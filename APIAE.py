import numpy as np
import tensorflow as tf
import pickle

def ldet(L):
    """
    Compute logdet(L), where L is lower triangular matrix (e.g. cholesky).
    L : (...,M,M)
    """
    Ldiag = tf.matrix_diag_part(L) # (...,M) 
    logLdiag = tf.log(1e-1+tf.abs(Ldiag)) # (...,M)
    ldet = tf.reduce_sum(logLdiag,axis=-1,keepdims=True) # (...,1)
    return tf.expand_dims(ldet,axis=-1) # (...,1,1)


class DynNet(object):
    """ Neural-net for the dynamics model: zdot=dz/dt=f(z) """
    def __init__(self, intermediate_units, n_z):
        self.intermediate_units = intermediate_units
        self.Layers = []

        # Construct the neural network
        for i, unit in enumerate(self.intermediate_units[:-1]):
            self.Layers.append(
                tf.layers.Dense(units=unit, activation=tf.nn.relu, name='DynLayer' + str(i)))  # fully-connected layer
        self.Layers.append(tf.layers.Dense(units=self.intermediate_units[-1], name='DynLayerLast'))

        # Below is for the later use of this network
        self.z_in = tf.placeholder(tf.float32, (None, n_z))
        self.zdot_out = self.compute_zdot(self.z_in)

        # Below is for the initialization
        self.zdot_ref = tf.placeholder(tf.float32, (None, n_z))
        self.loss = tf.reduce_mean(tf.reduce_sum((self.zdot_out - self.zdot_ref) ** 2, axis=-1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def compute_zdot(self, z):
        # Input :  z=(:,n_z)
        # Output: zdot=(:,n_z)
        z1dot = z[:, 1:2] # we fixed z_1 dot = z_2 for pendulum example
        z2dot = z # z_2 dot = neural_net(z)
        for layer in self.Layers:
            z2dot = layer(z2dot)
        zdot = tf.concat([z1dot, z2dot], axis=1)
        return zdot

    def initialize(self, sess, z_ref, zdot_ref, minibatchsize=500, training_epochs=5000, display_step=100):
        n_data = z_ref.shape[0]
        total_batch = int(n_data / minibatchsize)

        for epoch in range(training_epochs):
            avg_loss = 0
            nperm = np.random.permutation(n_data)

            # Loop over all batches
            for i in range(total_batch):
                minibatch_idx = nperm[i * minibatchsize:(i + 1) * minibatchsize]
                batch_zs = z_ref[minibatch_idx, :]
                batch_zdots = zdot_ref[minibatch_idx, :]

                opt, loss = sess.run((self.optimizer, self.loss), feed_dict={self.z_in: batch_zs, self.zdot_ref: batch_zdots})
                avg_loss += loss/total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))
                
        print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))
        
class GenNet(object):
    """ Neural-net for the generative model: x=g(z) """
    def __init__(self, intermediate_units, n_z):
        self.intermediate_units = intermediate_units
        self.Layers = []

        # Construct the neural network
        for i, unit in enumerate(self.intermediate_units[:-1]):
            self.Layers.append(tf.layers.Dense(units=unit, activation=tf.nn.relu,
                                               name='GenLayer' + str(i)))  # fully-connected layer w/ relu
        self.Layers.append(tf.layers.Dense(units=self.intermediate_units[-1], name='GenLayerLast'))  # last layer doesn't have relu activation

        # Below is for the later use of this network
        self.z_in = tf.placeholder(tf.float32, (None, n_z))
        self.x_unactivated = self.compute_x(self.z_in)
        self.x_out = tf.nn.sigmoid(self.x_unactivated)

    def compute_x(self, z):
        # Input : z=(:,n_z)
        # Output: x=(:,n_x)

        x = z[:, 0:1]
        for layer in self.Layers:
            x = layer(x)
        return x

class APIAE(object):
    """ APIAE for posterior inference in the latent space"""
    # Data shape : (Batch(B), Samples(L), Time(K), Dim1, Dim2)
    # If data is vector, Dim2 = 1
    
    def __init__(self,R,L,K,dt,n_x,n_z,n_u,ur,lr,isPlanner=False):   
        self.R = R # the number of improvements
        self.L = L # the number of trajectory sampled
        self.K = K # the number of time steps
        self.dt = dt # time interval
        self.sdt = np.sqrt(dt) # sqrt(dt)
        
        self.n_x = n_x # dimension of x; observation
        self.n_z = n_z # dimension of z; latent space
        self.n_u = n_u # dimension of u; control
        
        self.ur = ur # update rate
        self.lr = lr # learning rate
        
        self.isPlanner=isPlanner # flag whether this network is for the planning or not.
            
        self.xseq = tf.placeholder(tf.float32, shape=(None,1,self.K,self.n_x,1)) # input sequence of observations
        self.B = tf.shape(self.xseq)[0] # the number of batch
        
        # Define the networks for dynamics and generative model
        self.dynNet = DynNet(intermediate_units=[128,1], n_z=self.n_z) # dimension of the last layer is fixed as 1 for the pendulum example.
        self.genNet = GenNet(intermediate_units=[128,self.n_x], n_z=self.n_z)
        
        # Construct PI-net
        self._create_network()
        
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables and saver
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        
        # Launch the session
        self.sess.run(init)
    
    def _create_network(self):
        self.muhat_list = []
        self.museq_list = []
        self.Sighat_list = []
        self.Lhat_list = []
        self.Lhatinv_list = []
        self.Linvseq_list = []
        self.z0_list = [] 
        self.dwseq_list = []
        self.Kseq_list = []
        self.uffseq_list = []
        self.zseq_list = []
        self.S_list = []
        
        self._initial_state() # Define and initialize variables
        for r in range(self.R):
            self._Sampler() # Sampling z0(initial latent state) and dwseq(sequence of dynamic noise)
            self._Simulate() # Run dynamics and calculate the cost
            self._Update() # Update optimal control sequence & initial state dist.

    def _initial_state(self):
        # For initial states
        if self.isPlanner:
            self.mu0 = tf.placeholder(tf.float32, shape=(None,1,1,self.n_z,1)) # input sequence of observations
        else:
            self.mu0 = tf.zeros((1,1,1,self.n_z,1))
        self.Sig0 = tf.reshape(tf.diag([8.,10.]),(1,1,1,self.n_z,self.n_z)) # initialzie witth arbitrary value
        L0 = tf.cholesky(self.Sig0)
        self.L0inv = tf.matrix_inverse(L0) # cholesky lower triangular matrix of Sig0
        self.ldet_L0 = ldet(L0)
        
        muhat = tf.tile(self.mu0, (self.B,1,1,1,1))
        Sighat = tf.tile(self.Sig0, (self.B,1,1,1,1))
        Lhat = tf.cholesky(Sighat)
        
        self.Lhat_list.append(Lhat)
        self.Lhatinv_list.append(tf.matrix_inverse(Lhat))
        self.muhat_list.append(muhat)
        self.Sighat_list.append(Sighat)
        
        # For control input
        self.Kseq_list.append(tf.zeros((self.B,1,self.K-1,self.n_u,self.n_z))) # linear feedback term for control policy
        self.uffseq_list.append(tf.zeros((self.B,1,self.K-1,self.n_u,1))) # feed forward term for control policy
        
        self.museq_list.append(tf.zeros((self.B,1,self.K,self.n_z,1)))
        self.Linvseq_list.append(tf.eye(self.n_z, batch_shape=(self.B,1,self.K))) 
        
    def _Sampler(self):
        # For initial states
        muhat = self.muhat_list[-1]        
        if self.isPlanner:
            z0 = tf.tile(muhat,(1, self.L, 1,1,1))
        else:
            epsilon_z = tf.random_normal((self.B,self.L,1,self.n_z,1), 0., 1., dtype=tf.float32) # (B,L,1,n_z,1)
            Lhat_repeat = tf.tile(self.Lhat_list[-1], (1, self.L, 1,1,1))
            z0 = muhat + Lhat_repeat@epsilon_z
        self.z0_list.append(z0)   
            
        # For dynamic noise
        epsilon_u = tf.random_normal((self.B, self.L, self.K-1, self.n_u,1), 0., 1., dtype=tf.float32) # sample noise from N(0,1)
        dwseq = epsilon_u*self.sdt
        self.dwseq_list.append(dwseq)
        
    def _Simulate(self):
        # Load initial states and linear feedback parameters
        z0 = self.z0_list[-1]
        dwseq = self.dwseq_list[-1]
        
        museq = self.museq_list[-1]
        Linvseq = self.Linvseq_list[-1]
        Kseq = self.Kseq_list[-1]
        uffseq = self.uffseq_list[-1]

        # Reshape variables
        museq_repeat = tf.tile(museq, (1, self.L, 1, 1, 1))  # (B,L,K,n_z,1)
        Linvseq_repeat = tf.tile(Linvseq, (1,self.L,1,1,1)) # (B,L,K,n_z,n_z)
        Kseq_repeat = tf.tile(Kseq, (1,self.L,1,1,1)) # (B,L,K,n_u,n_z)
        uffseq_repeat = tf.tile(uffseq, (1,self.L,1,1,1)) # (B,L,K,n_u,1)
        
        museq_repeat_merge = tf.reshape(museq_repeat,(-1,self.K,self.n_z,1)) # (BL,K,n_z,1)
        Linvseq_repeat_merge = tf.reshape(Linvseq_repeat,(-1,self.K,self.n_z,self.n_z)) # (BL,K,n_z,n_z)
        Kseq_repeat_merge = tf.reshape(Kseq_repeat,(-1,self.K-1,self.n_u,self.n_z)) # (BL,K-1,n_u,n_z)
        uffseq_repeat_merge = tf.reshape(uffseq_repeat,(-1,self.K-1,self.n_u,1)) # (BL,K-1,n_u,1)
        dwseq_merge = tf.reshape(dwseq, (-1, self.K-1, self.n_u,1)) # (BL,K-1,n_u,1)

        z0_merge = tf.reshape(z0,(-1,self.n_z,1)) # (BL,n_z,1)
        zt_merge = z0_merge
        zt_merge_list = [zt_merge]
        utin_merge_list = []

        # Compute optimal control with standardized linear feedback policy
        for i in range(0,self.K-1):
            temp_val = Linvseq_repeat_merge[:,i,:,:]@(zt_merge-museq_repeat_merge[:,i,:,:])
            utin_merge = uffseq_repeat_merge[:,i,:,:] + Kseq_repeat_merge[:,i,:,:]@temp_val
            zt_merge = self.Propagate(zt_merge, utin_merge + dwseq_merge[:,i,:,:]/self.dt)
            zt_merge_list.append(zt_merge)
            utin_merge_list.append(utin_merge)

        # Reshape variables
        uinseq = tf.reshape(tf.stack(utin_merge_list, axis=1),(-1,self.L,self.K-1,self.n_u,1)) # (B,L,K-1,n_u,1)

        x_repeat_merge = tf.reshape(tf.tile(self.xseq, (1,self.L,1,1,1)), (-1, self.n_x, 1)) # (BLK,n_x,1)
        zseq = tf.reshape(tf.stack(zt_merge_list, axis=1),(-1,self.L,self.K,self.n_z,1)) # (B,L,K,n_z,1)
        zseq_merge = tf.reshape(zseq, (-1, self.n_z, 1)) # (BLK, n_x,1)

        #  Compute cost
        ss = self.state_cost(x_repeat_merge, zseq_merge) # state cost: (BLK,1,1)
        sc = self.control_cost(uinseq) # control cost: (B,L,1,1,1)
        
        if self.isPlanner:
            ss = self.K*tf.reshape(ss, (self.B,self.L,self.K,1,1))[:,:,-1:,:,:] # (B,L,1,1,1)
            self.S_list.append(ss + sc) #(B,L,1,1,1)
        else:
            ss = tf.reduce_sum(tf.reshape(ss, (self.B,self.L,self.K,1,1)), axis=2, keepdims=True) # (B,L,1,1,1)
            si = self.initial_cost() # Initial cost: (B,L,1,1,1)
            self.S_list.append(ss + sc + si) # (B,L,1,1,1)
        self.zseq_list.append(zseq) # (B,L,K,n_z,1)
        
    def _Update(self):
        # Load variables
        S = self.S_list[-1]  # (B,L,1,1,1)
        zseq = self.zseq_list[-1]  # (B,L,K,n_z,1)
        dwseq = self.dwseq_list[-1]  # (B,L,K-1,n_u,1)
        Kseq = self.Kseq_list[-1]  # (B,1,K-1,n_u,n_z)
        uffseq = self.uffseq_list[-1]  # (B,1,K-1,n_u,1)
        Linvseq_cur = self.Linvseq_list[-1][:,:,:-1,:,:]  # (B,L,K-1,n_z,n_z)
        museq_cur = self.museq_list[-1][:,:,:-1,:,:]  # (B,L,K-1,n_z,1)

        # Compute the weight, alpha = (B,L,1,1,1)
        Smin = tf.reduce_min(S, axis=1, keepdims=True)
        alpha = tf.exp(Smin-S) # to avoid positive large alpha
        norm = tf.reduce_sum(alpha, axis=1, keepdims=True)
        alpha = alpha/norm
        self.alpha_constant = tf.stop_gradient(alpha[:,:,0,0,0])  # clone alpha, but stop gradient

        # Compute mean and Cov. of L trajectories
        museq_next = tf.reduce_sum(alpha*zseq, axis=1, keepdims=True)  # (B,1,K,n_z,1)

        temp = (zseq - museq_next)  # (B,L,K,n_z,1)
        tempT = tf.transpose(temp, perm=(0, 1, 2, 4, 3))  # (B,L,K,1,n_z)
        offset = .01 * tf.eye(self.n_z, batch_shape=(1, 1, 1))
        Sigseq_next = offset + tf.reduce_sum(alpha*(temp@tempT), axis=1, keepdims=True)  # (B,1,K,n_z,n_z)

        Lseq_next = tf.cholesky(Sigseq_next)
        Linvseq_next = tf.matrix_inverse(Lseq_next)
        
        
        # Save variables
        self.museq_list.append(museq_next)
        self.Linvseq_list.append(Linvseq_next)
        self.muhat_list.append(museq_next[:, :, :1, :, :])
        self.Lhat_list.append(Lseq_next[:, :, :1, :, :])
        self.Lhatinv_list.append(Linvseq_next[:, :, :1, :, :])
        
        # Compute optimal control policy
        zseq = zseq[:, :, :-1, :, :]  # (B,L,K-1,n_z,1)
        museq_next = museq_next[:, :, :-1, :, :]  # (B,L,K-1,n_z,1)
        Lseq_next = Lseq_next[:, :, :-1, :, :]  # (B,L,K-1,n_z,n_z)
        Linvseq_next = Linvseq_next[:, :, :-1, :, :]  # (B,L,K-1,n_z,n_z)

        bstarseq = uffseq + Kseq @ (Linvseq_cur @ (museq_next - museq_cur)) + self.ur * tf.reduce_sum(alpha * dwseq,
                                                                                                    axis=1,
                                                                                                    keepdims=True) / self.dt  # (B,1,1,n_u,1)
        temp = tf.transpose(zseq - museq_next, perm=(0, 1, 2, 4, 3))
        LinvseqT = tf.transpose(Linvseq_next, perm=(0, 1, 2, 4, 3))
        astarseq = Kseq @ Linvseq_cur @ Lseq_next + self.ur * tf.reduce_sum(alpha * (dwseq @ temp), axis=1,
                                                                            keepdims=True) @ LinvseqT / self.dt

        # Save variables
        self.Kseq_list.append(astarseq)
        self.uffseq_list.append(bstarseq)
    
    def _create_loss_optimizer(self):
        S = tf.reduce_sum(self.alpha_constant*self.S_list[-1][:,:,0,0,0], axis = 1) # (B,L)
        self.cost_batch = tf.reduce_mean(S) # (scalar)
    
        # Use ADAM optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost_batch)
        
    def Propagate(self, zt, ut):
        """Simulate one-step forward"""
        # Input: zt=(BL,n_z,1), ut=(BL,n_u,1)
        # Output: znext=(BL,n_z,1)
        zt_squeeze = zt[:,:,0]
        zdott_squeeze = self.dynNet.compute_zdot(zt_squeeze)
        zdott = tf.expand_dims(zdott_squeeze, axis=-1)
        
        znext = zt + zdott*self.dt + tf.concat([tf.zeros_like(ut),ut],axis=1)*self.dt
        return znext
    
    def initial_cost(self): 
        """Compute the cost of initial state"""
        z0 = self.z0_list[-1] # (B,L,1,n_z,1) 
        mu0 = self.mu0 # (1,1,1,n_z,1) 
        L0inv = tf.tile(self.L0inv, (self.B,self.L,1,1,1)) # (B,L,1,n_z,n_z) 
        muhat = self.muhat_list[-1] # (B,1,1,n_z,1) 
        Lhatinv = tf.tile(self.Lhatinv_list[-1], (1,self.L,1,1,1)) # (B,L,1,n_z,n_z)    
        S0 = -(Lhatinv@(z0-muhat))**2 + (L0inv@(z0-mu0))**2 # (B,L,1,n_z,1)
        
        ldet_Lhat = ldet(self.Lhat_list[-1]) # (B,1,1,1,1)
        return 0.5 * tf.reduce_sum(S0,axis=-2, keepdims=True) + self.ldet_L0 - ldet_Lhat # (B,L,1,1,1) 

    def control_cost(self, uinseq):
        """Compute the cost of control input"""
        dwseq = self.dwseq_list[-1] 
        uTu = tf.reduce_sum(uinseq**2, axis=-2, keepdims=True) # (B,L,K,1,1) 
        uTdw = tf.reduce_sum(uinseq*dwseq, axis=-2, keepdims=True) # (B,L,K,1,1) 
        
        return tf.reduce_sum(uTu*0.5*self.dt + uTdw, axis=2, keepdims=True)
    
    def state_cost(self,xt_true,zt): # shape of inputs: (BLK, Dim, 1)
        """Compute the log-likelihood of observation xt given latent zt"""        
        xt = tf.expand_dims(self.genNet.compute_x(zt[:,:,0]),axis=-1)

        cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=xt_true, logits=xt), axis=1, keepdims=True)
        return cost*self.dt
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data, and return the cost of mini-batch."""
        opt, cost, zseq_list, museq_list = self.sess.run((self.optimizer, self.cost_batch, self.zseq_list,self.museq_list), 
                                  feed_dict={self.xseq: X})
        return cost, zseq_list, museq_list
    
    def saveWeights(self, filename="weights.pkl"):
        """Save the weights of neural networks"""
        weights = {}
        for i, layer in enumerate(self.dynNet.Layers):
            weights['d_w'+str(i)] = self.sess.run(layer.weights)

        for i, layer in enumerate(self.genNet.Layers):
            weights['g_w'+str(i)] = self.sess.run(layer.weights)    

        filehandler = open(filename,"wb")
        pickle.dump(weights,filehandler)
        filehandler.close()

        print('weight saved in '+filename)
        return weights

    def restoreWeights(self, filename="weights.pkl"):
        """Load the weights of neural networks"""
        filehandler = open(filename,"rb")
        weights = pickle.load(filehandler)
        filehandler.close()

        for i, layer in enumerate(self.dynNet.Layers):
            for j, w in enumerate(layer.weights):
                self.sess.run(tf.assign(w, weights['d_w'+str(i)][j]))

        for i, layer in enumerate(self.genNet.Layers):
            for j, w in enumerate(layer.weights):
                self.sess.run(tf.assign(w, weights['g_w'+str(i)][j]))  

        print('weight restored from '+filename)
        return weights