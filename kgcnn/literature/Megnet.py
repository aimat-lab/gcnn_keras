import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.casting import CastRaggedToDisjoint
from kgcnn.utils.activ import softplus2
from kgcnn.layers.disjoint.mlp import MLP

# Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals
# by Chi Chen, Weike Ye, Yunxing Zuo, Chen Zheng, and Shyue Ping Ong*
# https://github.com/materialsvirtuallab/megnet



class MEGnetBlock(ks.layers.Layer):
    """
    Megnet Block.
    
    Args:
        NodeEmbed (list, optional): List of node embedding dimension. Defaults to [16,16,16].
        EdgeEmbed (list, optional): List of edge embedding dimension. Defaults to [16,16,16].
        EnvEmbed (list, optional): List of environment embedding dimension. Defaults to [16,16,16].
        activation (func, optional): Activation function. Defaults to softplus2.
        use_bias (bool, optional): Use bias. Defaults to True.
        is_sorted (bool, optional): Edge index list is sorted. Defaults to True.
        has_unconnected (bool, optional): Has unconnected nodes. Defaults to False.
        **kwargs
    """
    
    def __init__(self,NodeEmbed=[16,16,16], 
                 EdgeEmbed=[16,16,16], 
                 EnvEmbed=[16,16,16] , 
                 activation=softplus2,
                 use_bias = True,
                 is_sorted = True,
                 has_unconnected=False,
                 **kwargs):
        """Initialize layer."""
        super(MEGnetBlock, self).__init__(**kwargs)
        self.NodeEmbed = NodeEmbed
        self.EdgeEmbed = EdgeEmbed
        self.EnvEmbed = EnvEmbed
        self.activation = activation
        self.use_bias = use_bias
        self.is_sorted = is_sorted
        self.has_unconnected = has_unconnected
        #Node
        self.lay_phi_n = ks.layers.Dense(self.NodeEmbed[0],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_n_1 = ks.layers.Dense(self.NodeEmbed[1],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_n_2 = ks.layers.Dense(self.NodeEmbed[2],activation='linear',use_bias=self.use_bias)
        self.lay_esum = PoolingEdgesPerNode(is_sorted=self.is_sorted,has_unconnected=self.has_unconnected)
        self.lay_gather_un = GatherState()
        self.lay_conc_nu = ks.layers.Concatenate(axis=-1)
        #Edge
        self.lay_phi_e = ks.layers.Dense(self.EdgeEmbed[0],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_e_1 = ks.layers.Dense(self.EdgeEmbed[1],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_e_2 = ks.layers.Dense(self.EdgeEmbed[2],activation='linear',use_bias=self.use_bias)
        self.lay_gather_n = GatherNodes()
        self.lay_gather_ue = GatherState()
        self.lay_conc_enu = ks.layers.Concatenate(axis=-1)
        #Environment
        self.lay_usum_e = PoolingAllEdges()
        self.lay_usum_n = PoolingNodes()
        self.lay_conc_u = ks.layers.Concatenate(axis=-1)
        self.lay_phi_u = ks.layers.Dense(self.EnvEmbed[0],activation=self.activation,use_bias=self.use_bias)        
        self.lay_phi_u_1 = ks.layers.Dense(self.EnvEmbed[1],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_u_2 = ks.layers.Dense(self.EnvEmbed[2],activation='linear',use_bias=self.use_bias)
        
    def build(self, input_shape):
        """Build layer."""
        super(MEGnetBlock, self).build(input_shape)
    def call(self, inputs):
        """Forward pass.
            
        Args:
            [node_input,edge_input,edge_index_input,env_input,len_node,len_edge]
        Returns:
            vp,ep,up
        """
        #Calculate edge Update
        node_input,edge_input,edge_index_input,env_input,len_node,len_edge = inputs
        e_n = self.lay_gather_n([node_input,len_node,edge_index_input,len_edge])
        e_u = self.lay_gather_ue([env_input,len_edge])
        ec = self.lay_conc_enu([e_n,edge_input,e_u])
        ep = self.lay_phi_e(ec) # Learning of Update Functions
        ep = self.lay_phi_e_1(ep) # Learning of Update Functions
        ep = self.lay_phi_e_2(ep) # Learning of Update Functions
        #Calculate Node update
        vb = self.lay_esum([node_input,len_node,ep,len_edge,edge_index_input]) # Summing for each node connections
        v_u = self.lay_gather_un([env_input,len_node])
        vc = self.lay_conc_nu([vb,node_input,v_u]) # Concatenate node features with new edge updates
        vp = self.lay_phi_n(vc) # Learning of Update Functions
        vp = self.lay_phi_n_1(vp) # Learning of Update Functions
        vp = self.lay_phi_n_2(vp) # Learning of Update Functions
        #Calculate environment update 
        es = self.lay_usum_e([ep,len_edge])
        vs = self.lay_usum_n([vp,len_node])
        ub = self.lay_conc_u([es,vs,env_input])
        up = self.lay_phi_u(ub)
        up = self.lay_phi_u_1(up)
        up = self.lay_phi_u_2(up) # Learning of Update Functions        
        return vp,ep,up




def getmodelMegnet(
                    # Input
                    input_node_shape,
                    input_edge_shape,
                    input_state_shape,
                    input_node_vocab = 95,
                    input_edge_vocab = 5,
                    input_state_vocab = 100,
                    input_node_embedd = 64,
                    input_edge_embedd = 64,
                    input_state_embedd = 64,
                    input_type = 'ragged', 
                    # Output
                    output_embedd = 'graph', #Only graph possible for megnet
                    output_use_bias = [True,True,True],
                    output_dim = [32,16,1],
                    output_activation = ['softplus2','softplus2','sigmoid'],
                    output_kernel_regularizer = [None,None,None],
                    output_activity_regularizer = [None,None,None],
                    output_bias_regularizer = [None,None,None],
                    output_type = 'padded',
                    #Model specs
                    is_sorted:bool = True,
                    has_unconnected:bool = False,
                    nblocks: int = 3,
                    n1: int = 64,
                    n2: int = 32,
                    n3: int = 16,
                    set2set_dim: int = 16,
                    use_bias = True,
                    act = 'softplus2',
                    l2_coef: float = None,
                    has_ff :bool = True,
                    dropout: float = None,
                    dropout_on_predict: bool = False,
                    use_set2set:bool = True,
                    npass: int = 3,
                    set2set_init:str = '0',
                    set2set_pool:str = "sum",
                    **kwargs):
    """
    Get Megnet model.
    
    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (,) embedding layer is used.
        input_node_vocab (int): Node input embedding vocabulary. Default is 95.
        input_edge_vocab (int): Edge input embedding vocabulary. Default is 5.
        input_state_vocab (int): State input embedding vocabulary. Default is 100.
        input_node_embedd (int): Node input embedding dimension. Default is 64.
        input_edge_embedd (int): Edge input embedding dimension. Default is 64.
        input_state_embedd (int): State embedding dimension. Default is 64.
        input_type (str): Specify input type. Only 'ragged' is supported. 
        
        output_embedd (str): Graph or node embedding of the graph network. Default is 'graph'.
        output_use_bias (bool,list): Use bias for output. Optionally list for multiple layer. Defautl is [True,True,True].
        output_dim (list): Output dimension. Optionally list for multiple layer. Default is [32,16,1].
        output_activation (list): Activation function. Optionally list for multiple layer. Defautl is ['softplus2','softplus2','sigmoid'].
        output_kernel_regularizer (list): Kernel regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_activity_regularizer (list): Activity regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_bias_regularizer (list): Bias regularizer for output. Optionally list for multiple layer. Defautl is [None,None,None].
        output_type (str): Tensor output type. Default is 'padded'.
        
        is_sorted (bool): Are edge indices sorted. Default is True.
        has_unconnected (bool): Has unconnected nodes. Default is False.
        nblocks (int): Number of block. Default is 3.
        n1 (int): n1 parameter. Default is 64.
        n2 (int): n2 parameter. Default is 32.
        n3 (int): n3 parameter. Default is 16.
        set2set_dim (int): Set2set dimension. Default is 16.
        use_bias (bools): Use bias. Default is True.
        act (func): Activation function. Default is softplus2.
        l2_coef (float): Regularization coefficient. Default is None.
        has_ff (bool): Feed forward layer. Default is True.
        dropout (float): Use dropout. Default is None.
        dropout_on_predict (bool): Use dropout on prediction. Default is False.
        use_set2set (bool): Use set2set. Default isTrue.
        npass (int): Set2Set iterations. Default is 3.
        set2set_init (str): Initialize method. Default is '0'.
        set2set_pool (str): Pooling method in set2set. Default is "sum".
    
    Returns:
        model (tf.keras.models.Model): MEGnet model.
    """
    
    if(len(input_node_shape) == 1):
        node_input = ks.layers.Input(shape=input_node_shape,name='node_input',dtype ="float32",ragged=True)
        n =  ks.layers.Embedding(input_node_vocab, input_node_embedd , name='node_embedding')(node_input)
    else:
        node_input = ks.layers.Input(shape=input_node_shape,name='node_input',dtype ="float32",ragged=True)
        n = node_input
        
    if(len(input_edge_shape)== 1):
        edge_input = ks.layers.Input(shape=input_edge_shape,name='edge_input',dtype ="float32",ragged=True)
        ed = ks.layers.Embedding(input_edge_vocab, input_state_embedd , name='edge_embedding')(edge_input)
    else:
        edge_input = ks.layers.Input(shape=input_edge_shape,name='edge_input',dtype ="float32",ragged=True)
        ed = edge_input
        
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    
    if(len(input_state_shape) == 0):
        env_input = ks.Input(shape=input_state_shape, dtype='float32' ,name='state_input')
        uenv = ks.layers.Embedding(input_state_vocab,input_state_embedd, name='state_embedding')(env_input)
    else:
        env_input = ks.Input(shape=input_state_shape, dtype='float32' ,name='state_input')
        uenv = env_input

    
    n,node_len,ed,edge_len,edi = CastRaggedToDisjoint()([n,ed,edge_index_input])
   
      
    dropout_training = True if dropout_on_predict else None
    if l2_coef is not None:
        reg = ks.regularizers.l2(l2_coef)
    else:
        reg = None
    
    if(isinstance(act,str)):
        if(act == 'softplus2'):
            act = softplus2
    
    #starting 
    vp = n
    up = uenv
    ep = ed
    vp = ks.layers.Dense(n1,activation=act)(vp)
    vp = ks.layers.Dense(n2,activation=act)(vp)
    ep = ks.layers.Dense(n1,activation=act)(ep)
    ep = ks.layers.Dense(n2,activation=act)(ep)
    up = ks.layers.Dense(n1,activation=act)(up)
    up = ks.layers.Dense(n2,activation=act)(up)
    ep2 = ep
    vp2 = vp
    up2 = up
    for i in range(0,nblocks):
        if(has_ff == True and i>0):
            vp2 = ks.layers.Dense(n1,activation=act)(vp)
            vp2 = ks.layers.Dense(n2,activation=act)(vp2)
            ep2 = ks.layers.Dense(n1,activation=act)(ep)
            ep2 = ks.layers.Dense(n2,activation=act)(ep2)
            up2 = ks.layers.Dense(n1,activation=act)(up)
            up2 = ks.layers.Dense(n2,activation=act)(up2)
            
        #MEGnetBlock
        vp2,ep2,up2 = MEGnetBlock(NodeEmbed=[n1,n1,n2],
                                  EdgeEmbed=[n1,n1,n2],
                                  EnvEmbed=[n1,n1,n2],
                                  activation=act,
                                  is_sorted = is_sorted,
                                  has_unconnected = has_unconnected,
                                  name='megnet_%d'%i)([vp2,ep2,edi,up2,node_len,edge_len])
        # skip connection
        
        if dropout:
            vp2 = ks.layers.Dropout(dropout, name='dropout_atom_%d'%i)(vp2, training=dropout_training)
            ep2 = ks.layers.Dropout(dropout, name='dropout_bond_%d'%i)(ep2, training=dropout_training)
            up2 = ks.layers.Dropout(dropout, name='dropout_state_%d'%i)(up2, training=dropout_training)
        
        vp = ks.layers.Add()([vp2, vp])
        up = ks.layers.Add()([up2, up])
        ep = ks.layers.Add()([ep2 ,ep])

    if(use_set2set == True):
        vp = ks.layers.Dense(set2set_dim,activation='linear')(vp)
        ep = ks.layers.Dense(set2set_dim,activation='linear')(ep)
        vp = Set2Set(set2set_dim,T=npass,pooling_method=set2set_pool,init_qstar = set2set_init)([vp,node_len])
        ep = Set2Set(set2set_dim,T=npass,pooling_method=set2set_pool,init_qstar = set2set_init)([ep,edge_len])
    else:
        vp = PoolingNodes()([vp,node_len])
        ep = PoolingAllEdges()([ep,edge_len])
    
    ep =  ks.layers.Flatten()(ep)
    vp =  ks.layers.Flatten()(vp)
    final_vec = ks.layers.Concatenate(axis=-1)([ vp, ep,up])
    
    
    if dropout:
        final_vec = ks.layers.Dropout(dropout, name='dropout_final')(final_vec, training=dropout_training)
    
    # final dense layers 
    main_output = MLP(output_dim,
                    mlp_use_bias = output_use_bias,
                    mlp_activation = output_activation,
                    mlp_activity_regularizer=output_kernel_regularizer,
                    mlp_kernel_regularizer=output_kernel_regularizer,
                    mlp_bias_regularizer=output_bias_regularizer)(final_vec)     
        
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input,env_input], outputs=main_output)   

    return model




