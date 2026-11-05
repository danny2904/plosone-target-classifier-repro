import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel, AutoModel

class XLMRMultiTask(RobertaPreTrainedModel):
    def __init__(self, config, use_fusion=True):
        super().__init__(config)
        self.use_fusion = use_fusion
        
        # Load pre-trained Backbone (Shared Encoder)
        self.roberta = AutoModel.from_config(config)
        
        # Classification Head (Sentence Level)
        # Input: [CLS] token embedding OR [CLS] + Rationale
        input_dim = config.hidden_size * 2 if use_fusion else config.hidden_size
        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(input_dim, config.num_labels)
        
        # Token Classification Head (Word/Span Level)
        # Input: All token embeddings
        self.token_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Output 2 classes per token: 0 (Normal), 1 (Toxic)
        self.token_classifier = nn.Linear(config.hidden_size, 3) 
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,       # Classification labels (Clean/Hate)
        token_labels=None, # Token-level labels (0/1 for each token)
        alpha=1.0,          # Weight for Token Loss
        use_consistency=True,
        lambda_const=0.1,
        cls_class_weights=None,
        cls_sample_weights=None,
        token_class_weights=None,
        token_focal_gamma=0.0,
        token_dice_lambda=0.0,
        token_sample_weights=None,
    ):
        
        # By default, PhoBERT (Roberta) doesn't use token_type_ids, but we keep arg for compatibility
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]  # (Batch, Seq_Len, Hidden)
        pooled_output = outputs[1]    # (Batch, Hidden) - [CLS] token representation
        token_mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else torch.ones(
            sequence_output.size(0), sequence_output.size(1), 1, device=sequence_output.device
        )

        # --- Task 2 Pre-calc: Token Labeling for Rationale Extraction ---
        token_output = self.token_dropout(sequence_output)
        token_logits = self.token_classifier(token_output) # (Batch, Seq_Len, 2)
        
        # --- Explanation-Guided Fusion ---
        if self.use_fusion:
            # Calculate Toxic Probabilities (Softmax) -> Use as Attention Weights
            token_probs = torch.softmax(token_logits, dim=-1) # (Batch, Seq_Len, 3)
            # Prob of being toxic = Prob(B-Toxic) + Prob(I-Toxic)
            toxic_probs = (token_probs[:, :, 1] + token_probs[:, :, 2]).unsqueeze(-1)  # (Batch, Seq_Len, 1)
            toxic_probs = toxic_probs * token_mask
            
            # Weighted Average of Token Embeddings based on Toxicity (Rationale Vector)
            sum_toxic = torch.sum(toxic_probs, dim=1) + 1e-9
            rationale_vector = torch.sum(sequence_output * toxic_probs, dim=1) / sum_toxic # (Batch, Hidden)
            
            # Concatenate [CLS] + Rationale Vector
            combined_features = torch.cat((pooled_output, rationale_vector), dim=1)
            cls_input = combined_features
        else:
            # No Fusion (Vanilla MTL)
            cls_input = pooled_output
            
        # --- Task 1: Classification ---
        cls_output = self.cls_dropout(cls_input)
        cls_logits = self.cls_classifier(cls_output) # (Batch, Num_Labels)

        total_loss = None
        
        if labels is not None and token_labels is not None:
            if cls_class_weights is not None or cls_sample_weights is not None:
                loss_fct_cls = nn.CrossEntropyLoss(weight=cls_class_weights, reduction="none")
                cls_losses = loss_fct_cls(cls_logits.view(-1, self.config.num_labels), labels.view(-1))
                if cls_sample_weights is not None:
                    w = cls_sample_weights.view(-1).to(cls_losses.device).float()
                    cls_losses = cls_losses * w
                loss_cls = cls_losses.mean()
            else:
                loss_fct_cls = nn.CrossEntropyLoss()
                loss_cls = loss_fct_cls(cls_logits.view(-1, self.config.num_labels), labels.view(-1))
            
            # Span-first token loss:
            # - optional class weighting
            # - optional focal modulation
            # - optional per-sample weighting (for hard-mined samples)
            # - optional dice regularizer on toxic mask (B+I)
            token_ce = nn.CrossEntropyLoss(
                ignore_index=-100, weight=token_class_weights, reduction="none"
            )(token_logits.view(-1, 3), token_labels.view(-1)).view(token_labels.size(0), token_labels.size(1))
            valid_mask = (token_labels != -100).float()
            per_tok_loss = token_ce

            if token_focal_gamma and float(token_focal_gamma) > 0:
                probs = torch.softmax(token_logits, dim=-1)
                clamped_labels = token_labels.clamp(min=0)
                pt = torch.gather(probs, dim=-1, index=clamped_labels.unsqueeze(-1)).squeeze(-1)
                pt = torch.where(valid_mask > 0, pt, torch.ones_like(pt))
                focal_factor = (1.0 - pt).pow(float(token_focal_gamma))
                per_tok_loss = per_tok_loss * focal_factor

            per_sample_token_loss = (per_tok_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
            if token_sample_weights is not None:
                tsw = token_sample_weights.view(-1).to(per_sample_token_loss.device).float()
                per_sample_token_loss = per_sample_token_loss * tsw
            loss_token = per_sample_token_loss.mean()
            
            # Weighted Sum Multi-task Loss
            total_loss = loss_cls + (alpha * loss_token)

            if token_dice_lambda and float(token_dice_lambda) > 0:
                probs = torch.softmax(token_logits, dim=-1)
                pred_toxic = (probs[:, :, 1] + probs[:, :, 2]) * valid_mask
                gold_toxic = (token_labels > 0).float() * valid_mask
                eps = 1e-8
                inter = 2.0 * torch.sum(pred_toxic * gold_toxic)
                denom = torch.sum(pred_toxic) + torch.sum(gold_toxic) + eps
                dice_loss = 1.0 - (inter + eps) / denom
                total_loss = total_loss + float(token_dice_lambda) * dice_loss
            
            # --- Consistency/Alignment Loss ---
            if use_consistency and lambda_const > 0:
                # Re-calculate probs if not done by fusion block (could optimize but keeping clean logic)
                token_probs = torch.softmax(token_logits, dim=-1) # (Batch, Seq_Len, 3)
                toxic_probs = (token_probs[:, :, 1] + token_probs[:, :, 2]).unsqueeze(-1)  # (Batch, Seq_Len, 1)
                toxic_probs = toxic_probs * token_mask

                clean_mask = (labels == 0)
                if clean_mask.sum() > 0:
                    clean_toxic_probs = toxic_probs[clean_mask].squeeze(-1) # (Num_Clean, Seq_Len)
                    clean_token_mask = token_mask[clean_mask].squeeze(-1)
                    denom = torch.sum(clean_token_mask).clamp_min(1.0)
                    loss_consistency = torch.sum(clean_toxic_probs) / denom
                    total_loss += (lambda_const * loss_consistency)
            
        return {
            'loss': total_loss,
            'cls_logits': cls_logits,
            'token_logits': token_logits
        }
