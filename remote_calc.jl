import Pkg
	Pkg.activate()

using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, PlutoUI, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels, Distributed

struct KGNN <: AbstractGraphLayer
    A_2 #Adjacency matrix of hidden graph
    X_2 #Node features of hidden graph
    num_edge_types  #Number of edge labels
    p #Walk length hyperparameter
    σ
end

# Layer constructor with random initializations
# Number of edge types does not include de-edges
KGNNLayer(num_nodes, num_node_types, num_edge_types, p) = KGNN(
    Float32.(rand(num_nodes, num_nodes, num_edge_types+1)), 
    Float32.(rand(num_nodes,num_node_types)), 
    Int(num_edge_types), 
    Int(p), 
    relu
)

@functor KGNN

# normalized random walk layer, output ranges from [0,1]
function (l::KGNN)(A, X)::Float32

    nv = size(l.A_2)[1] # number of vertices
    
    n_edge_types = size(l.A_2)[3] # number of edge labels

    A_norm_mx = Zygote.@ignore upper_triang(nv)

    # Deletions of the self loops and copying upper triangle to the lower triangle to make symmetric. The final product resembles the "split adjacency"
    A_2_herm = stack([relu.((l.A_2[:,:,i].*A_norm_mx)+(l.A_2[:,:,i].*A_norm_mx)') for i ∈ 1:n_edge_types])

    id = Matrix{Float32}(I, nv, nv) |> (isa(l.A_2, CuArray) ? gpu : cpu) # identity matrix on cpu or fpu depending on which is being used
    
    A_2_adj = stack([sum([A_2_herm[:,:,i] for i ∈ 1:n_edge_types]).+id for i ∈ 1:n_edge_types]) # a slice of this matrix houses the sum of an edge label vector for each edge. the identity matrix is added to avoid dividing by 0 on the main daigonal. The matrix is copied for each slice to match dimensions of A_2_herm
    
    A_2_norm = A_2_herm#./A_2_adj # result - all edge feature vectors sum to 1

    # same result as edge features except for node features
    X_2_norm = relu.(l.X_2)#./sum([l.X_2[:,i] for i ∈ 1:l.num_edge_types])
    

    # inner product normalization - k(x,y)/(k(x,x)*k(y,y))^.5
    return sum((vec(X*X_2_norm')*vec(X*X_2_norm')'.*sum([kron(A_2_norm[:,:,i],A[:,:,i]) for i ∈ 1:l.num_edge_types]))^l.p)/(sum((vec(X*X')*vec(X*X')'.*sum([kron(A[:,:,i],A[:,:,i]) for i ∈ 1:l.num_edge_types]))^l.p)*sum((vec(X_2_norm*X_2_norm')*vec(X_2_norm*X_2_norm')'.*sum([kron(A_2_norm[:,:,i],A_2_norm[:,:,i]) for i ∈ 1:l.num_edge_types]))^l.p))^.5

end

function (l::KGNN)(fg::AbstractFeaturedGraph)
    return l(global_feature(fg), node_feature(fg)')
end

function upper_triang(sz) # generates an n x n matrix where the upper triangle is 1 and the lower triangle, and main diagonal is 0
	mx = zeros(Float32,sz,sz)
	for i ∈ 1:sz
		for j ∈ 1:sz
			if j>i
				mx[i,j] = 1
			end
		end
	end
	return mx
end

function remote_calc(model,tuple)
	
    x, y = tuple

    loss, grads = Flux.withgradient(model) do m
    
        y_hat = m(x)
        Flux.mse(y_hat, y)
        
    end
    return (;loss, grads)
end

function make_kgnn(graph_size,n_node_types_model,n_edge_types_model,p,n_hidden_graphs)
	model = Chain(
		Parallel(vcat, 
			[KGNNLayer(graph_size,n_node_types_model,n_edge_types_model,p) for i ∈ 1:n_hidden_graphs]...
		),
		Dense(n_hidden_graphs => 4, tanh),
	    Dense(4 => 4, tanh),
		#Dropout(0.2),
		Dense(4 => 2, tanh),
		softmax,
	)
	return model
end