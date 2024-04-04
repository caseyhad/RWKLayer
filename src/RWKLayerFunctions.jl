

module RWKLayerFunctions
using MetaGraphs, GeometricFlux, Functors, Zygote, ProgressMeter, CSV, DataFrames, CUDA, MolecularGraphKernels, MolecularGraph, Graphs, Flux, Random, LinearAlgebra, MLUtils
export KGNN, find_labels, train_kgnn, mg_to_fg, hidden_graph_view, hg_to_mg


function kron2d(A, B)
    A4 = reshape(A, 1, size(A, 1), 1, size(A, 2))
    B4 = reshape(B, size(B, 1), 1, size(B, 2), 1)
    C4 = A4 .* B4
    C = reshape(C4, size(A, 1) * size(B, 1), size(A, 2) * size(B, 2))
end


function split_adjacency(g::MetaGraph,edge_labels)::Array{Float32}
    # Constructs an array of size n x n x l+1, where n is the number of nodes and l is the number of edge labels. each slice of the array is an adjaceny matrix of only edges of a specific label. The last slice of the array is reserved for the non-bonding pairs of vertice. Summing through the slices returns the adjacency matrix of a densly connected graph with no self-loops. 
    
    nv = size(g)[1] # number of vertices
    nt = length(edge_labels) # number of edge types  
    adj_layers = zeros(Float32,nv, nv, nt+1) # the extra level at the bottom is the non-edges (1-A)

	adj_layers[:,:, end] .= 1
	adj_layers[:,:, end] .-= Matrix{Float32}(I, nv, nv)
    # check each edge for a label matching the current slice of the split adjacency matrix
    for l ∈ eachindex(edge_labels)
        for edge ∈ edges(g)
            if get_prop(g, edge, :label) == edge_labels[l]
                # add the edge to the matrix and delete the non-edge from the last layer
                adj_layers[src(edge),dst(edge),l] = 1.0

				adj_layers[src(edge),dst(edge),end] = 0
				adj_layers[dst(edge),src(edge),end] = 0
            end
        end
        # make symmetric
        adj_layers[:,:,l] = adj_layers[:,:,l]+adj_layers[:,:,l]'
    end
    
    return adj_layers
end
# converts R^2 adjacency matrix into an R^3 array where each layer is an adjacency matrix for only one edge feature
function split_adjacency(fg::AbstractFeaturedGraph)::Array{Float32}
    ef = fg.ef.signal # edge features from featuredGraph
    nv = size(fg.nf.signal)[2]
    ne = size(ef)[2]
    nt = size(ef)[1] # number of edge types
    edge_array = [edge for edge ∈ edges(fg)][1:ne]
    adj_layers = zeros(Float32, nv, nv, nt+1)

	adj_layers[:,:, end] .= 1
	adj_layers[:,:, end] .-= Matrix{Float32}(I, nv, nv)
    for edge_idx ∈ eachindex(edge_array)
        v₁,v₂ = edge_array[edge_idx][2]

        adj_layers[v₁,v₂,1:nt] .= ef[:,edge_idx]
        adj_layers[v₂,v₁,1:nt] .= ef[:,edge_idx]

		adj_layers[v₁,v₂,end] = 0
        adj_layers[v₂,v₁,end] = 0
    end
    return adj_layers
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


struct KGNN <: AbstractGraphLayer
    h_adj # A djacency matrix of hidden graph
    h_nf # Number of node features
    n_ef  # Number of edge features
    p # Walk length hyperparameter
    a # activation function for graph features, sigmoid/relu both work equally
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
function (l::KGNN)(A, X)#::Float32

	#number of vertices in the hidden graph
    nv = size(l.h_adj)[1]

	#upper triangle matrix with diag 0
    adj_reg_mx = Zygote.@ignore upper_triang(nv)|> (isa(l.h_adj, CuArray) ? gpu : cpu)

	#store the identity matrix on either the gpu or cpu depending on which is in use
	id = Matrix{Float32}(I, nv, nv) |> (isa(l.h_adj, CuArray) ? gpu : cpu)

	#make each layer a square matrix by copying its upper triangle to the lower half
    h_adj_sqr = stack(map(adj -> adj.*adj_reg_mx + (adj.*adj_reg_mx)'.+id, eachslice(l.h_adj, dims = 3)))
    
	#apply the softmax function to all edge feature vectors
	h_adj_r = permutedims(softmax(permutedims(h_adj_sqr ,(3,2,1))),(3,2,1))

	#apply the softmax function to all node feature vectors
    h_nf_r = stack(softmax.(eachrow(l.h_nf)))'

	#random walk kernel calculation between input graph and hidden graph
    k_xy = sum((vec(X*h_nf_r')*vec(X*h_nf_r')'.*sum([kron2d(h_adj_r[:,:,i],A[:,:,i]) for i ∈ 1:l.n_ef]))^l.p)
    
	#random walk between input graph and input graph
    k_xx = sum((vec(X*X')*vec(X*X')'.*sum([kron2d(A[:,:,i],A[:,:,i]) for i ∈ 1:l.n_ef]))^l.p)
        
	#random walk between hidden graph and hidden graph
    k_yy = sum((vec(h_nf_r*h_nf_r')*vec(h_nf_r*h_nf_r')'.*sum([kron2d(h_adj_r[:,:,i],h_adj_r[:,:,i]) for i ∈ 1:l.n_ef]))^l.p)

	#cosine norm maps the output to the range [0,1]
    k_xy/(k_xx*k_yy)^.5
	

end

function (l::KGNN)(fg::AbstractFeaturedGraph)
    return l(global_feature(fg), node_feature(fg)')
end


## algo to add NamedTuple Gradients together for gradient accumulation in batch learning. Credit: Adrian Henle
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

function hidden_graph_view(model, graph_number)
	number_hg = length(model.layers[1][:])
	h_adj = model.layers[1][graph_number].h_adj
	h_nf = model.layers[1][graph_number].h_nf
	n_ef = model.layers[1][graph_number].n_ef
	a = model.layers[1][graph_number].a
	
	nv = size(h_adj)[1] # number of vertices

    adj_reg_mx = Zygote.@ignore upper_triang(nv)|> (isa(h_adj, CuArray) ? gpu : cpu) # upper triangle matrix with diag 0

	id = Matrix{Float32}(I, nv, nv) |> (isa(h_adj, CuArray) ? gpu : cpu)

    h_adj_sqr = stack([(h_adj[:,:,i].*adj_reg_mx)+(h_adj[:,:,i].*adj_reg_mx)'.+id for i ∈ 1:size(h_adj)[3]]) # make each layer a square matrix by copying its upper triangle to the lower half
    
	h_adj_r = permutedims(softmax(permutedims(h_adj_sqr ,(3,2,1))),(3,2,1))

    h_nf_r = stack(softmax.(eachrow(h_nf)))'

	graph_feature = Float32.([i == graph_number for i ∈ 1:number_hg])
	res = Chain(model.layers[3:end]...)(graph_feature)

	return h_adj_r, h_nf_r, res
end

function gk_contribution_map(res,G::MetaGraph;dynamic_range = 1::Real)
	#store the trained model on device being used (gpu or cpu)
	investigated_model = res.output_model |> device

	#initialize a dict for the appearence counts of the node in graphlets
	vertex_counts = Dict(zip(vertices(G), zeros(length(vertices(G)))))
	#initialize a dict for the accumulated probability of the graphlets (that contain v) of being positive class
	vertex_outcomes = Dict(zip(vertices(G), zeros(length(vertices(G)))))

	#generate each all-connected size-k subgraphs of G
	graphlets = MolecularGraphKernels.con_sub_g(res.output_model.layers[1][1].p,G)

	#make a vector of graphlets as metagraphs
	graph_vec = [induced_subgraph(G,graphlet)[1] for graphlet ∈ graphlets]

	#convert to featuredGraphs
	featuredgraph_vec = mg_to_fg(graph_vec,res.data.labels.edge_labels,res.data.labels.vertex_labels)

	#pass graphlets through the model
	model_outs = [investigated_model(featuredgraph|>device) for featuredgraph ∈ featuredgraph_vec]|>cpu

	for i ∈ eachindex(model_outs)
		model_prediction = model_outs[i]
		#list of vertices in current graphlet
		vertices_present = graphlets[i]
		for v ∈ vertices_present
			vertex_counts[v] +=1
			#add the prob. of positive class, subract the prob. of the negative class
			vertex_outcomes[v] += (model_prediction[1]-model_prediction[2])
			
		end
	end
	for i ∈ eachindex(vertex_outcomes)
		vertex_outcomes[i] = vertex_outcomes[i]/(vertex_counts[i]*dynamic_range)
	end
	return vertex_outcomes
end

begin
	function viz_node_colors(weights)::Vector{RGBA}
		wts_vec = -[((weights[i])+1)/2 for i in eachindex(weights)].+1
	return [RGBA(1-i, 0, i) for i in wts_vec]
	
	end
	function viz_node_colors(g::MetaGraph)
    	return viz_node_colors(gk_contribution_map(res_med, g, dynamic_range = .462))
	end
	
end

function hg_to_mg(
	res,graph_number;
	edge_threshold=.2::Real,
	vertex_threshold=0::Real,
)
	model = res.output_model

	col_to_label = Dict(zip(1:length(res.data.labels.vertex_labels), res.data.labels.vertex_labels))

	slice_to_label = Dict(zip(1:length(res.data.labels.edge_labels), res.data.labels.edge_labels))
	
	h_adj_r, h_nf_r, class_pred = hidden_graph_view(model, graph_number)
	# get adjacency matrix by edge thresholding
	adj = sum([h_adj_r[:,:,i].>edge_threshold for i in axes(h_adj_r[:,:,1:end-1], 3)])

	v_props = []
	vertex_deletions = []
	# set node features, delete vertices (and edges to them) that are below threshold value
	for r in 1:size(h_nf_r)[1]
		if all(h_nf_r[r,:].<vertex_threshold)
			push!(vertex_deletions, r)
		else
			push!(v_props,col_to_label[argmax(h_nf_r[r,:])])
		end
	end
	A_idx = [i for i ∈ 1:size(adj)[1] if i ∉ vertex_deletions]
	A = adj[A_idx, A_idx]

	# generate graph topology
	g = MetaGraph(SimpleGraph(A))

	for v in vertices(g)
		set_prop!(g, v, :label, v_props[v])
	end

	# set edge weights
	wts = reshape(
		[slice_to_label[x[3]] for x in argmax(h_adj_r[:,:,1:end-1]; dims=3)], 
		axes(h_adj_r[:, :, 1])
	)

	alphas = reshape(
		[x for x in maximum(h_adj_r[:,:,1:end-1]; dims=3)], 
		axes(h_adj_r[:, :, 1])
	)
	for e in edges(g)
		set_prop!(g, e, :label, wts[src(e), dst(e)])
		set_prop!(g, e, :alpha, alphas[src(e), dst(e)])
	end
	
	return g
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

function batch_graphs(batch)
	graphs = []
	classes = []

	for (x,y) ∈ batch
		push!(graphs,x)
		push!(classes,y)
	end
	
	nf = hcat([fg.nf.signal for fg ∈ graphs]...)
	ef = hcat([fg.ef.signal for fg ∈ graphs]...)
	final_size = size(nf)[2]
	n_ef = size(ef)[1]

	Ã = zeros(final_size,final_size,n_ef+1)
	n_ct = 1
	for graph ∈ graphs
		nv = size(graph.nf.signal)[2]
		Ã[n_ct:n_ct+nv-1,n_ct:n_ct+nv-1,:] = graph.gf.signal
		n_ct += nv
	end

	adjm = sum([Ã[:,:,o] for o ∈ 1:size(Ã)[3]-1])
	return (FeaturedGraph(adjm; nf=nf, ef=ef, gf = Ã), mean(stack(classes),dims = 2))
end

function make_kgnn(graph_size,n_node_types_model,n_edge_types_model,p,n_hidden_graphs,device)
	model = Chain(
		Parallel(vcat, 
			[KGNNLayer(graph_size,n_node_types_model,n_edge_types_model,p) for i ∈ 1:n_hidden_graphs]...
		),device,
		#Dropout(.2),
		#Dense(n_hidden_graphs => n_hidden_graphs, relu),
		Dense(n_hidden_graphs => n_hidden_graphs, tanh),
		Dense(n_hidden_graphs => 2, σ),
		softmax
	)
	return model
end

function train_from_fmap(class_labels, graph_vector, test_classes, test_graphs, model, batch_sz::Int, n::Int, lr, device)

	n_samples = length(graph_vector)
	new_model = Chain(model.layers[3:end]...) |> device
	feature_map = model.layers[1] |> device

	input_mx = stack(feature_map.(graph_vector |> device)) |> cpu

	test_mx = stack(feature_map.(test_graphs |> device)) |> cpu

	oh_class = Flux.onehotbatch(class_labels, [true, false])
	data_class = Flux.DataLoader((input_mx, oh_class), batchsize=batch_sz, shuffle=true) |> device

	optim = Flux.setup(Flux.Adam(lr), new_model)

	losses = []
	test_accuracy = []
	test_recall = []

	for epoch in 1:n

		epoch_loss = 0

		for batch ∈ data_class

			X,Y = batch

			loss, grads = Flux.withgradient(new_model) do m

				Ŷ = m(X)
				Flux.crossentropy(Ŷ, Y)
				
			end
			
			Flux.update!(optim, new_model, grads[1]|>device)
			epoch_loss += loss/batch_sz
			
		end
		preds = new_model.(eachslice(stack(test_mx), dims = 2)|>gpu)|>cpu
		pred_vector = reduce(hcat,preds)
		pred_bool = [pred_vector[1,i].==maximum(pred_vector[:,i]) for i ∈ 1:size(pred_vector)[2]]

		tp = sum(pred_bool.==test_classes.==1)
		fp = sum(pred_bool.==test_classes.+1)
		fn = sum(pred_bool.==test_classes.-1)
		
		epoch_acc = sum(pred_bool.==test_classes)/length(pred_bool)
		epoch_rec = tp/(tp+fn)

		push!(losses, epoch_loss)
		push!(test_accuracy, epoch_acc)
		push!(test_recall, epoch_rec)

	end
	return losses, new_model, test_accuracy, test_recall
end

function train_model_batch_dist(class_labels, graph_vector, test_classes, test_graphs, model, batch_sz::Int, n::Int, lr, device)
	#good values: n = 900, lr = .1

	n_samples = length(graph_vector)
	oh_class = Flux.onehotbatch(class_labels, [true, false])
	data_class = zip(graph_vector, [oh_class[:,i] for i ∈ 1:n_samples])

	optim = Flux.setup(Flux.Adam(lr), model)

	losses = []
	test_accuracy = []
	test_recall = []

	@showprogress for epoch in 1:n

		batches = Base.Iterators.partition(shuffle(collect(data_class)), batch_sz) |> device
		epoch_loss = 0
		
		for batch ∈ batches

            results = map(batch) do (x,y)
                loss, grads = Flux.withgradient(model) do m
    
			        ŷ = m(x)
			        Flux.crossentropy(ŷ, y)
			        
			    end
			    return (;loss, grads)
			end
			
			epoch_loss += sum([results[i].loss for i ∈ eachindex(results)])
			grads_s = [results[i].grads for i ∈ eachindex(results)]

			
			
			∇ = reduce(add_gradients, grads_s|>cpu)
			

			Flux.update!(optim, model, ∇[1]|>device)
		end
		preds = model.(test_graphs|>device)|>cpu
		pred_vector = reduce(hcat,preds)
		pred_bool = [pred_vector[1,i].==maximum(pred_vector[:,i]) for i ∈ 1:size(pred_vector)[2]]

		tp = sum(pred_bool.==test_classes.==1)
		fp = sum(pred_bool.==test_classes.+1)
		fn = sum(pred_bool.==test_classes.-1)
		
		epoch_acc = sum(pred_bool.==test_classes)/length(pred_bool)
		epoch_rec = tp/(tp+fn)

		push!(losses, epoch_loss/n_samples)
		push!(test_accuracy, epoch_acc)
		push!(test_recall, epoch_rec)

	end
	return losses, model, test_accuracy, test_recall
end

function train_model_graph_batch(class_labels, graph_vector, model, batch_sz, n, lr,)
	#good values: n = 900, lr = .1

	n_samples = length(graph_vector)
	oh_class = Flux.onehotbatch(class_labels, [true, false])
	data_class = zip(graph_vector, [oh_class[:,i] for i ∈ 1:n_samples])

	optim = Flux.setup(Flux.Adam(lr/n_samples), model)
	
	losses = []

	for epoch in 1:n

		batches = Base.Iterators.partition(shuffle(collect(data_class)), batch_sz)
		
		for batch ∈ batches

			(x,y) = batch_graphs(batch) |> device
			
			loss, grads = Flux.withgradient(model) do m

				y_hat = m(x)
				Flux.crossentropy(y_hat, y)
				
			end
			push!(losses, loss)
			Flux.update!(optim, model, grads[1]|>device)
		end
        
	end
	return losses, model
end

function train_kgnn(
	graph_data::Vector, 
	class_data::Vector;
	lr = 0.05,
	n_epoch = 600::Int,
	p = 5::Int,
	n_hg = 8::Int,
	size_hg = 6::Int,
	device = gpu,
	batch_sz = 1,
	train_portion = 0.80,
	two_stage = false
)
	if length(graph_data)!=length(class_data)
		error("Graph vector and class vector must be the same size")
	end
	
	class_labels = []
	meta_graph_vector = Vector{MetaGraphs.MetaGraph{Int64, Float64}}()
	if typeof(graph_data) == Vector{MolecularGraph.graphmol}
		for i ∈ eachindex(graph_data)
			try
				push!(meta_graph_vector,MetaGraph(graph_data[i]))
				push!(class_labels,class_data[i])
			catch e
				@warn  "Some MolecularGraph objects could not be converted to MetaGraph"
			end
		end
	end

	if typeof(graph_data) == Vector{String}
		for i ∈ eachindex(graph_data)
			smiles_string = graph_data[i]
			try
				mol = smilestomol(smiles_string)
				push!(meta_graph_vector,MetaGraph(mol))
				push!(class_labels,class_data[i])
			catch e
				@warn  "Some SMILES could not be converted to MetaGraph"
			end
		end
	end

	if typeof(graph_data) == Vector{MetaGraphs.MetaGraph{Int64, Float64}}
		meta_graph_vector = graph_data
		class_labels = class_data
	end

	if length(meta_graph_vector)==0
		error("No MetaGraphs could be converted from input graph data")
	end

	labels = find_labels(meta_graph_vector)

	featured_graphs = mg_to_fg(
		meta_graph_vector,
		labels.edge_labels,
		labels.vertex_labels
	)

	n_node_types = length(labels.vertex_labels)
	n_edge_types = length(labels.edge_labels)

	train_set_sz = Int(round(train_portion*length(featured_graphs)))
	test_set_sz = length(featured_graphs)-train_set_sz

	shuffled_data = shuffle(collect(zip(featured_graphs,class_labels)))
	
	training_data = shuffled_data[1:train_set_sz]
	
	testing_data = shuffled_data[train_set_sz+1:min(train_set_sz+test_set_sz+1,size(shuffled_data)[1])]
	
	training_graphs = getindex.(training_data,1)
	training_classes = getindex.(training_data,2)

	testing_graphs = getindex.(testing_data,1)
	testing_classes = getindex.(testing_data,2)

	if two_stage
		model1 = make_kgnn(
			size_hg,
			n_node_types,
			n_edge_types,
			p,
			n_hg,
			device
			)|>device


		losses_fm, trained_model_fm, epoch_test_accuracy_fm, epoch_test_recall_fm = train_model_batch_dist(
			training_classes, 
			training_graphs,
			testing_classes,
			testing_graphs,
			model1,
			batch_sz,
			n_epoch,
			lr,
			device
		)



		losses_nm, new_classifier, test_accuracy_nm, test_recall_nm = train_from_fmap(
			training_classes, 
			training_graphs,
			testing_classes,
			testing_graphs,
			trained_model_fm, 
			32, 
			n_epoch, 
			lr, 
			device)

		losses = vcat(losses_fm, losses_nm)

		epoch_test_accuracy = vcat(epoch_test_accuracy_fm, test_accuracy_nm)

		epoch_test_recall = vcat(epoch_test_recall_fm, test_recall_nm)

		output_model = cpu(Chain(trained_model_fm.layers[1:2]...,new_classifier.layers...))

		data = (;training_data, testing_data, labels)

		inputs = (lr, batch_sz)
		
		return (;losses, output_model, epoch_test_accuracy, epoch_test_recall, data, inputs)

	else
		model1 = make_kgnn(
			size_hg,
			n_node_types,
			n_edge_types,
			p,
			n_hg,
			device
			)|>device


		losses, trained_model, epoch_test_accuracy, epoch_test_recall = train_model_batch_dist(
			training_classes, 
			training_graphs,
			testing_classes,
			testing_graphs,
			model1,
			batch_sz,
			n_epoch,
			lr,
			device
		)

		output_model = cpu(trained_model)

		data = (;training_data, testing_data, labels)

		inputs = (lr, batch_sz)
		
		return (;losses, output_model, epoch_test_accuracy, epoch_test_recall, data, inputs)
	end
end

end