import torch
import torch.nn as nn

# Definition du modèle : prédit à la fois l'item et l'event
class GRUItemEventModel(nn.Module):
    def __init__(self, num_items, num_events, embedding_dim, hidden_size):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.event_embedding = nn.Embedding(num_events, embedding_dim)
        self.gru = nn.GRU(embedding_dim * 2, hidden_size, batch_first=True)
        self.item_fc = nn.Linear(hidden_size, num_items)
        self.event_fc = nn.Linear(hidden_size, num_events)

    def forward(self, item_seq, event_seq):
        item_emb = self.item_embedding(item_seq)
        event_emb = self.event_embedding(event_seq)
        x = torch.cat([item_emb, event_emb], dim=2)
        out, _ = self.gru(x)
        out = out.contiguous().view(-1, out.size(2))
        item_out = self.item_fc(out)
        event_out = self.event_fc(out)
        return item_out.view(item_seq.size(0), item_seq.size(1), -1), \
               event_out.view(item_seq.size(0), item_seq.size(1), -1)