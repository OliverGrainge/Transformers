import torch 
import torch.nn as nn 
import torch.nn.functional as F



class ImageTokenizer(nn.Module):
    def __init__(self, args):
        super(ImageTokenizer, self).__init__()

        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.num_patches = (args.img_size // args.patch_size) ** 2

        self.projection = nn.Linear(args.patch_size * args.patch_size * args.in_channels, args.embed_size)
        self.position_embedding = nn.Parameter(torch.randn(self.num_patches + 1, args.embed_size))

        self.cls_token = nn.Parameter(torch.randn(1, 1, args.embed_size))

    def forward(self, images):
        # Size check
        assert images.shape[-1] == self.img_size and images.shape[-2] == self.img_size, \
            f"Images must be of size {self.img_size}x{self.img_size}"

        # Split images into patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        batch_size, h, w, c, ph, pw = patches.shape
        patches = patches.reshape(batch_size, h * w, c * ph * pw)

        # Embed patches
        x = self.projection(patches)

        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.position_embedding

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention_out, _ = self.attention(query, key, value)
        x = self.norm1(attention_out + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention_out, _ = self.attention(query, key, value)
        x = self.norm1(attention_out + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.num_patches = (args.img_size // args.patch_size) ** 2
        
        self.patch_embed = nn.Linear(args.patch_size * args.patch_size * args.in_channels, args.embed_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, args.embed_size))

        self.tokenizer = ImageTokenizer(args)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, args.embed_size))
        self.dropout = nn.Dropout(args.dropout)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(args.embed_size, args.heads, args.dropout, args.forward_expansion)
            for _ in range(args.depth)
        ])
        
        self.classifier_head = nn.Linear(args.embed_size, args.num_classes)

    def forward(self, x):
        # Patchify and embed
        
        #x = self.patch_embed(x)
        #cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        #x = torch.cat([cls_tokens, x], dim=1)
        #x += self.position_embedding
        x = self.tokenizer(x)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, x, x)
        
        # Classifier head
        cls_token_final = x[:, 0]
        out = self.classifier_head(cls_token_final)
        
        return out








