import Pkg
	Pkg.activate()
	using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, PlutoUI, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels, Distributed, BenchmarkTools

function split_adjacency(g::MetaGraph,edge_labels)::Array{Float32}
    # Constructs an array of size n x n x l+1, where n is the number of nodes and l is the number of edge labels. each slice of the array is an adjaceny matrix of only edges of a specific label. The last slice of the array is reserved for the non-bonding pairs of vertice. Summing through the slices returns the adjacency matrix of a densly connected graph with no self-loops. 
    
    nv = size(g)[1] # number of vertices
    nt = length(edge_labels) # number of edge types  
    adj_layers = zeros(Float32,nv, nv, nt+1) # the extra level at the bottom is the non-edges (1-A)
    adj_layers[:,:,nt+1] = ones(nv,nv).-Matrix{Float32}(I, nv, nv)

    # check each edge for a label matching the current slice of the split adjacency matrix
    for l ∈ eachindex(edge_labels)
        for edge ∈ edges(g)
            if get_prop(g, edge, :label) == edge_labels[l]
                # add the edge to the matrix and delete the non-edge from the last layer
                adj_layers[src(edge),dst(edge),l] = 1.0
                adj_layers[src(edge),dst(edge),nt+1] = 0
                adj_layers[dst(edge),src(edge),nt+1] = 0
            end
        end
        # make symmetric
        adj_layers[:,:,l] = adj_layers[:,:,l]+adj_layers[:,:,l]'
    end
    
    return adj_layers
end
# converts R^2 adjacency matrix into an R^3 array where each layer is an adjacency matrix for only one edge feature
function split_adjacency(fg::AbstractFeaturedGraph)::Array{Float32}
    ef = fg.ef.signal
    nv = size(fg.nf.signal)[2]
    ne = size(ef)[2]
    nt = size(ef)[1]
    edge_array = [edge for edge ∈ edges(fg)][1:ne]
    adj_layers = zeros(Float32, nv, nv, nt+1)
    adj_layers[:,:,nt+1] = ones(nv,nv).-Matrix{Float32}(I, nv, nv)
    for edge_idx ∈ eachindex(edge_array)
        v₁,v₂ = edge_array[edge_idx][2]

        adj_layers[v₁,v₂,1:nt] .= ef[:,edge_idx]
        adj_layers[v₂,v₁,1:nt] .= ef[:,edge_idx]
        
        adj_layers[v₂,v₁,nt+1] = 0
        adj_layers[v₁,v₂,nt+1] = 0
    end
    return adj_layers
end

module LayerDefinition

using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, PlutoUI, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels, Distributed, BenchmarkTools
export make_kgnn, remote_calc, KGNN, l, upper_triang


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



end

function recursive_addition!(a, b)
    if isa(a, Number)
        return a + b
    elseif isnothing(a)
        return b
    end
    
    for key in keys(a)
        x = recursive_addition!(a[key], b[key])
        if isa(x, Number)
            a[key] = x
        end
    end
    
    return a
end

function add_gradients(grad1, grad2)
    g1, g2 = (x -> x[1]).([grad1, grad2])   
    return Tuple((; layers=recursive_addition!(deepcopy(g1), g2)))
end

function mg_to_fg(meta_graphs::Vector{MetaGraphs.MetaGraph{Int64, Float64}}, edge_labels::Vector, node_labels::Vector)::Array{AbstractFeaturedGraph} # converts graphs from MetaGraph to the AbstractFeaturedGraphs type used by GeometricFlux. Edges all originate from the smaller indexed vertex to the larger indexed vertex (Edge: 1 => 3, not 3 => 1). Edge features are ordered based on 1) smallest index origin vertex and 2) smallest terminating vertex (same as reading the adjacency matrix right to left, top to bottom). output is one-hot style for edge/node features
	return_vec = []
	for graph ∈ meta_graphs
		nv = size(vertices(graph))[1]
		edge_list = collect(edges(graph))
		ne = size(edge_list)[1]
		nf = zeros(Float32,nv,length(node_labels))
		ef = zeros(Float32,ne,length(edge_labels))
		adj = zeros(Float32,nv,nv)
		for v ∈ vertices(graph)
			nf[v,:] = props(graph,v)[:label] .== node_labels
		end
		for i ∈ 1:ne
			e = edge_list[i]
			ef[i,:] = props(graph,e)[:label] .== edge_labels
		end
		for i ∈ 1:nv
			for j ∈ 1:nv
				if props(graph,i,j) != Dict()
					adj[i,j] = 1
				end
			end
		end
		adj = adj + adj'
		Ã = split_adjacency(graph,edge_labels)
		fg = FeaturedGraph(adj; nf=nf', ef=ef', gf = Ã)
		push!(return_vec,fg)
	end
	return return_vec
end

function find_labels(graph_vector)
	vertex_labels = []
	edge_labels = []
	for graph ∈ graph_vector
		for v ∈ vertices(graph)
			push!(vertex_labels,props(graph,v)[:label])
		end
		for e ∈ edges(graph)
			push!(edge_labels,props(graph,e)[:label])
		end
	end
	return (vertex_labels = sort(unique(vertex_labels)),edge_labels = sort(unique(edge_labels)))
end

function train_model_batch_dist(class_labels, graph_vector, model, batch_sz, n, lr,)
	#good values: n = 900, lr = .1

	n_samples = length(graph_vector)
	oh_class = Flux.onehotbatch(class_labels, [true, false])
	data_class = zip(graph_vector, [oh_class[:,i] for i ∈ 1:n_samples])

	optim = Flux.setup(Flux.Adam(lr/n_samples), model)
	
	losses = []

	for epoch in 1:n

		batches = Base.Iterators.partition(shuffle(collect(data_class)), 3)
		epoch_loss = 0
		
		for batch ∈ batches
			

            results = pmap(batch) do a
                remote_calc(model,a)
			end
			
			epoch_loss += sum([results[i].loss for i ∈ eachindex(results)])
			grads_s = [results[i].grads for i ∈ eachindex(results)]
			
			∇ = reduce(add_gradients, grads_s)

			Flux.update!(optim, model, ∇[1])
		end

		push!(losses, epoch_loss/n_samples)
	end
	return losses, model
end

##end of function definitions##


@everywhere using .LayerDefinition

data = CSV.read(Base.download("https://github.com/SimonEnsemble/graph-kernel-SVM-for-toxicity-of-pesticides-to-bees/raw/main/BeeToxAI%20Data/File%20S1%20Acute%20contact%20toxicity%20dataset%20for%20classification.csv"),DataFrame)

btx_class_labels = [data[i,:Outcome] == "Toxic" for i ∈ 1:length(data[:,:Outcome])]


errored_smiles = []
btx_graphs = Vector{MetaGraphs.MetaGraph{Int64, Float64}}()
disallowed_features = ['.','+']
for z ∈ 1:length(data[!,:SMILES])
    smiles_string = data[z,:SMILES]
    try
        mol = smilestomol(smiles_string)
        push!(btx_graphs,MetaGraph(mol))
    catch e
        push!(errored_smiles, [z,smiles_string])
    end
end

labels = find_labels(btx_graphs)

btx_featuredgraphs = mg_to_fg(btx_graphs,labels.edge_labels,labels.vertex_labels)

graph_classes = btx_class_labels[[i ∉[errored_smiles[i][1] for i ∈ 1:length(errored_smiles)] for i in 1:length(btx_class_labels)]]

btx_losses,btx_model = train_model_batch_dist(graph_classes[1:40],btx_featuredgraphs[1:40],make_kgnn(8,12,4,4,10), 6, 10,.1);

@save pwd()*"\\btx_losses.jld2" btx_losses
@save pwd()*"\\btx_model.jld2" btx_model



