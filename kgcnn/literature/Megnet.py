"""@package: Model for the MEGnet as defined by Chen et al. 2019
https://doi.org/10.1021/acs.chemmater.9b01294

"""

import tensorflow.keras as ks
import tensorflow as tf

from kgcnn.layers.disjoint.gather import GatherState,GatherNodesIngoing,GatherNodesOutgoing,GatherNodes
from kgcnn.layers.disjoint.conv import ConvFlatten
from kgcnn.layers.disjoint.pooling import PoolingEdgesPerNode,PoolingNodes,PoolingAllEdges
from kgcnn.layers.disjoint.set2set import Set2Set
from kgcnn.layers.disjoint.batch import RaggedToDisjoint,CastListToRagged,CastRaggedToList,CorrectIndexListForSubGraph



def softplus2(x):
    """
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow
    Args:
        x: (Tensor) input tensor
    Returns:
         (Tensor) output tensor
    """
    return ks.backend.relu(x) + ks.backend.log(0.5*ks.backend.exp(-ks.backend.abs(x)) + 0.5)


class MEGnetBlock(ks.layers.Layer):
    """
    Layer for the MEGnet block
    """
    def __init__(self,NodeEmbed=[16,16,16], EdgeEmbed=[16,16,16], EnvEmbed=[16,16,16] , activation=softplus2,use_bias = True ,**kwargs):
        super(MEGnetBlock, self).__init__(**kwargs)
        self.NodeEmbed = NodeEmbed
        self.EdgeEmbed = EdgeEmbed
        self.EnvEmbed = EnvEmbed
        self.activation = activation
        self.use_bias = use_bias
        #Node
        self.lay_phi_n = ConvFlatten(self.NodeEmbed[0],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_n_1 = ConvFlatten(self.NodeEmbed[1],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_n_2 = ConvFlatten(self.NodeEmbed[2],activation='linear',use_bias=self.use_bias)
        self.lay_esum = PoolingEdgesPerNode()
        self.lay_gather_un = GatherState()
        self.lay_conc_nu = ks.layers.Concatenate(axis=-1)
        #Edge
        self.lay_phi_e = ConvFlatten(self.EdgeEmbed[0],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_e_1 = ConvFlatten(self.EdgeEmbed[1],activation=self.activation,use_bias=self.use_bias)
        self.lay_phi_e_2 = ConvFlatten(self.EdgeEmbed[2],activation='linear',use_bias=self.use_bias)
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
        super(MEGnetBlock, self).build(input_shape)
    def call(self, inputs):
        #Calculate edge Update
        node_input,edge_input,edge_index_input,env_input,len_node,len_edge = inputs
        e_n = self.lay_gather_n([node_input,edge_index_input])
        e_u = self.lay_gather_ue([env_input,len_edge])
        ec = self.lay_conc_enu([e_n,edge_input,e_u])
        ep = self.lay_phi_e(ec) # Learning of Update Functions
        ep = self.lay_phi_e_1(ep) # Learning of Update Functions
        ep = self.lay_phi_e_2(ep) # Learning of Update Functions
        #Calculate Node update
        vb = self.lay_esum([ep,edge_index_input]) # Summing for each node connections
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
    def compute_output_shape(self, input_shape):
        is_node,is_edge,is_edge_index,is_env,_,_ = input_shape
        return((is_node[0],self.NodeEmbed),(is_edge[0],self.EdgeEmbed),(is_env[0],self.EnvEmbed))



def getmodelMegnet(  
                input_type = "raggged",
                nfeat_edge: int = None,
                nfeat_global: int = None,
                nfeat_node: int = None,
                nblocks: int = 3,
                n1: int = 64,
                n2: int = 32,
                n3: int = 16,
                set2set_dim: int = 16,
                nvocal: int = 95,
                embedding_dim: int = 16,
                nbvocal: int = None,
                bond_embedding_dim: int = None,
                ngvocal: int = None,
                global_embedding_dim: int = None,
                ntarget: int = 1,
                use_bias = True,
                act = softplus2,
                l2_coef: float = None,
                has_ff :bool = True,
                dropout: float = None,
                dropout_on_predict: bool = False,
                is_classification: bool = False,
                use_set2set = True,
                npass: int = 3,
                set2set_init = '0',
                set2set_pool = "sum",
                **kwargs
                  ):

    # Inputs

    if nfeat_node is None:
        node_input =  ks.Input(shape=(None,),dtype='int32', name='atom_int_input',ragged=True)  # only z as feature
        n =  ks.layers.Embedding(nvocal, embedding_dim, name='atom_embedding')(node_input)
    else:
        node_input = ks.Input(shape=(None,nfeat_node), dtype='float32' ,name='atom_feature_input',ragged=True)
        n = node_input
    if nfeat_edge is None:
        edge_input =  ks.Input(shape=(None,), dtype='int32', name='bond_int_input',ragged=True)
        ed =  ks.layers.Embedding(nbvocal, bond_embedding_dim, name='bond_embedding')(edge_input)
    else:
        edge_input = ks.Input(shape=(None,nfeat_edge), dtype='float32' ,name='bond_feature_input',ragged=True)
        ed = edge_input
    if nfeat_global is None:
        env_input =  ks.Input(shape=(), dtype='int32', name='state_int_input')
        uenv = ks.layers.Embedding(ngvocal, global_embedding_dim, name='state_embedding')(env_input)
    else:
        env_input = ks.Input(shape=(nfeat_global,), dtype='float32' ,name='state_feature_input')
        uenv = env_input
    
    edge_index_input = ks.layers.Input(shape=(None,2),name='edge_index_input',dtype ="int64",ragged=True)
    
    
    n,node_len,ed,edge_len,edi = RaggedToDisjoint()([n,ed,edge_index_input])
   
      
    # Get the setting for the training kwarg of Dropout
    dropout_training = True if dropout_on_predict else None
    if l2_coef is not None:
        reg = ks.regularizers.l2(l2_coef)
    else:
        reg = None
        
    #starting 
    vp = n
    up = uenv
    ep = ed
    vp = ConvFlatten(n1,activation=act)(vp)
    vp = ConvFlatten(n2,activation=act)(vp)
    ep = ConvFlatten(n1,activation=act)(ep)
    ep = ConvFlatten(n2,activation=act)(ep)
    up = ConvFlatten(n1,activation=act)(up)
    up = ConvFlatten(n2,activation=act)(up)
    ep2 = ep
    vp2 = vp
    up2 = up
    for i in range(0,nblocks):
        if(has_ff == True and i>0):
                vp2 = ConvFlatten(n1,activation=act)(vp)
                vp2 = ConvFlatten(n2,activation=act)(vp2)
                ep2 = ConvFlatten(n1,activation=act)(ep)
                ep2 = ConvFlatten(n2,activation=act)(ep2)
                up2 = ConvFlatten(n1,activation=act)(up)
                up2 = ConvFlatten(n2,activation=act)(up2)
            
        #MEGnetBlock
        vp2,ep2,up2 = MEGnetBlock(NodeEmbed=[n1,n1,n2],EdgeEmbed=[n1,n1,n2],EnvEmbed=[n1,n1,n2],activation=act,name='megnet_%d'%i)([vp2,ep2,edi,up2,node_len,edge_len])
        # skip connection
        
        if dropout:
            vp2 = ks.layers.Dropout(dropout, name='dropout_atom_%d'%i)(vp2, training=dropout_training)
            ep2 = ks.layers.Dropout(dropout, name='dropout_bond_%d'%i)(ep2, training=dropout_training)
            up2 = ks.layers.Dropout(dropout, name='dropout_state_%d'%i)(up2, training=dropout_training)
        
        vp = ks.layers.Add()([vp2, vp])
        up = ks.layers.Add()([up2, up])
        ep = ks.layers.Add()([ep2 ,ep])

    if(use_set2set == True):
        vp = ConvFlatten(set2set_dim,activation='linear')(vp)
        ep = ConvFlatten(set2set_dim,activation='linear')(ep)
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
    final_vec = ks.layers.Dense(n2, activation=act, kernel_regularizer=reg, name='readout_0')(final_vec)
    final_vec = ks.layers.Dense(n3, activation=act, kernel_regularizer=reg, name='readout_1')(final_vec)
    
    if is_classification:
        final_act = 'sigmoid'
    else:
        final_act = None
        
    main_output = ks.layers.Dense(ntarget, activation=final_act, name='readout_2')(final_vec)
    
    
    model = ks.models.Model(inputs=[node_input,edge_input,edge_index_input,env_input], outputs=main_output)   

    return model




