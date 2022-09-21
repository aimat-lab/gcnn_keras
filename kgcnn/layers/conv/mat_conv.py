import tensorflow as tf
from einops import rearrange
ks = tf.keras
from tensorflow.keras.layers import Layer

def exists(val):
    return val is not None

def default(val, d):
    return d if not exists(val) else val

class Attention(Layer):
    '''
    MAT Mutli Head Attention to inject Adj and Distance effects
    Initial version without mask and fix kernel ie exp(-D) more logical
    strongly inspired by lucidbrains MAT source code
    '''
    def __init__(self,
                 units=64,
                 heads=8,
                 La = 1,
                 Lg = 0.5,
                 Ld = 0.5,
                 **kwargs):
        """Initialize layer."""
        super(Attention, self).__init__(**kwargs)
        self.heads = int(heads)
        self.units = int(units)
        self.La = La
        self.Lg = Lg
        self.Ld = Ld
        self.innerdim = self.heads*self.units
        self.to_qkv = ks.layers.Dense(self.innerdim*3,use_bias=False)
        self.scale =  self.units ** -0.5

    def build(self, input_shape):
        """Build layer."""
        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        node, adj, dist, node_mask, adj_mask, dist_mask = inputs
        print(node.shape, adj.shape, dist.shape)
        print(node_mask.shape, adj_mask.shape, dist_mask.shape)
        # convert node input into qkv matrix
        qkv = self.to_qkv(node)
        # rearrange and unstack the matrix to get each elements 
        q,k,v = tf.unstack(rearrange(qkv, 'b n (h qkv d) -> b h n qkv d', h = self.heads, qkv = 3),axis=-2)
        # compute attention coefficients
        dots = tf.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # apply the softmax on the dots product matrix
        # TODO: must add mask to be compatible to softmax transformation of pad zeros to pad zeros!
        attn = ks.activations.softmax(dots, axis=-1)
        # MAT part adding the dist and adj need to add a extra axis for the final sum of the attns
        # TODO: must add mask to be compatible to Attention transformation of "distance pad inf to distance pad inf!"
        # based on the article : dist and adj are defined by rdkit functions GetAdjacencyMatrix & GetDistanceMatrix 
        # so [Batch, Natom x Natom ] matrix for both.
        dist = rearrange(dist, 'b i j -> b () i j')
        adj = rearrange(adj, 'b i j -> b () i j')
        # equation 2 in MAT

        attn  = attn*self.La + adj*self.Lg + tf.exp(-dist)*self.Ld
        out = tf.einsum('b h i j, b h j d -> b h i d', attn, v)
        # reshape to the correct dims
        out = rearrange(out, 'b h n d -> b n (h d)')
        # go back to the input shape
        out = ks.layers.Dense(node.shape[-1])(out)
        return out

    def get_config(self):
        """Update layer config."""
        config = super(Attention, self).get_config()
        config.update({"units": self.units,"heads": self.heads,"La": self.La,
                      "Lg": self.Lg,"Ld": self.Ld})
        return config

class FF(Layer):
    '''
    FeedForward MLP  
    strongly inspired by lucidbrains MAT source code
    '''
    def __init__(self,
                 dim_out = None,
                 mult = 4,
                 **kwargs):
        """Initialize layer."""
        super(FF, self).__init__(**kwargs)
        self.mult = mult
        self.dim_out = dim_out
    def build(self, input_shape):
        """Build layer."""
        super(FF, self).build(input_shape)

    def call(self, x, **kwargs):
        dim = x.shape[-1]
        dim_out = default(self.dim_out, dim)
        x = ks.layers.Dense(dim*self.mult)(x)
        x = ks.activations.gelu(x)
        x = ks.layers.Dense(dim_out)(x)
        return x
        
    def get_config(self):
        """Update layer config."""
        config = super(FF, self).get_config()
        config.update({"mult": self.mult,"dim_out": self.dim_out})
        return config
