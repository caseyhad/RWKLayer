### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ def5bc20-8835-4484-82ca-1cee86d9a34e
begin
	import Pkg
	Pkg.activate()
	using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, PlutoUI, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels, BenchmarkTools, Distributed
	TableOfContents(title="Random walk layer w/ edge labels")
end

# ╔═╡ b01bcf77-efb2-41db-bd16-dd285ba090e0
device = gpu

# ╔═╡ ae30e674-5e45-461a-9249-33ac79e44fd6
function kron2d(A, B)
    A4 = reshape(A, 1, size(A, 1), 1, size(A, 2))
    B4 = reshape(B, size(B, 1), 1, size(B, 2), 1)
    C4 = A4 .* B4
    C = reshape(C4, size(A, 1) * size(B, 1), size(A, 2) * size(B, 2))
end

# ╔═╡ 522c0377-b1f5-4768-9db8-9a7bec01311a
begin
	function split_adjacency(g::MetaGraph,edge_labels)::Array{Float32}
		# Constructs an array of size n x n x l+1, where n is the number of nodes and l is the number of edge labels. each slice of the array is an adjaceny matrix of only edges of a specific label. The last slice of the array is reserved for the non-bonding pairs of vertice. Summing through the slices returns the adjacency matrix of a densly connected graph with no self-loops. 
		
		nv = size(g)[1] # number of vertices
		nt = length(edge_labels) # number of edge types  
		adj_layers = zeros(Float32,nv, nv, nt) # the extra level at the bottom is the non-edges (1-A)

		# check each edge for a label matching the current slice of the split adjacency matrix
		for l ∈ eachindex(edge_labels)
			for edge ∈ edges(g)
				if get_prop(g, edge, :label) == edge_labels[l]
					# add the edge to the matrix and delete the non-edge from the last layer
					adj_layers[src(edge),dst(edge),l] = 1.0
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
		adj_layers = zeros(Float32, nv, nv, nt)
		for edge_idx ∈ eachindex(edge_array)
			v₁,v₂ = edge_array[edge_idx][2]

			adj_layers[v₁,v₂,1:nt] .= ef[:,edge_idx]
			adj_layers[v₂,v₁,1:nt] .= ef[:,edge_idx]
		end
		return adj_layers
	end
end

# ╔═╡ 10cdfcd0-d2e7-4ca6-a113-f944b1ddb99c
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

# ╔═╡ 1bf932f8-05c6-4237-962c-9e99c5c29004
begin
	struct KGNN <: AbstractGraphLayer
	    h_adj # A djacency matrix of hidden graph
		h_nf # Number of node features
		n_ef  # Number of edge features
		p # Walk length hyperparameter
	    σ # activation function for graph features, sigmoid/relu both work equally
	end

	# Layer constructor with random initializations
	# Number of edge types does not include de-edges
	KGNNLayer(num_nodes, num_node_types, num_edge_types, p) = KGNN(
		Float32.(rand(num_nodes, num_nodes, num_edge_types)), 
		Float32.(rand(num_nodes,num_node_types)), 
		Int(num_edge_types), 
		Int(p), 
		σ
	)
	
	@functor KGNN

	# normalized random walk layer, output ranges from [0,1]
	function (l::KGNN)(A, X)::Float32

		nv = size(l.h_adj)[1] # number of vertices

		adj_reg_mx = Zygote.@ignore upper_triang(nv)|> (isa(l.h_adj, CuArray) ? gpu : cpu) # upper triangle matrix with diag 0

		h_adj_r = stack([l.σ.((l.h_adj[:,:,i].*adj_reg_mx)+(l.h_adj[:,:,i].*adj_reg_mx)') for i ∈ 1:l.n_ef]) # make each layer a square matrix, apply activation
		
		h_nf_r = l.σ.(l.h_nf)
		
		k_xy = sum((vec(X*h_nf_r')*vec(X*h_nf_r')'.*sum([kron2d(h_adj_r[:,:,i],A[:,:,i]) for i ∈ 1:l.n_ef]))^l.p) # random walk kernel calculation between input and hidden
		
		k_xx = sum((vec(X*X')*vec(X*X')'.*sum([kron2d(A[:,:,i],A[:,:,i]) for i ∈ 1:l.n_ef]))^l.p) # " input and input
			
		k_yy = sum((vec(h_nf_r*h_nf_r')*vec(h_nf_r*h_nf_r')'.*sum([kron2d(h_adj_r[:,:,i],h_adj_r[:,:,i]) for i ∈ 1:l.n_ef]))^l.p) # " hidden and hidden

		return k_xy/(k_xx*k_yy)^.5 # inner product normalization result bounded [0,1]
	
	end
	
	function (l::KGNN)(fg::AbstractFeaturedGraph)
	    return l(global_feature(fg), node_feature(fg)')
	end
end

# ╔═╡ 48fb7ca7-0acc-4c42-95d4-45f22ecd3817
begin ## functions to add NamedTuple Gradients together for gradient accumulation in batch learning. Credit: Adrian Henle
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
end

# ╔═╡ 4ef939f4-9e6e-4ae0-96c3-0331afcfd195
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

# ╔═╡ 3658ade9-e6ec-4445-8ea7-51bf51c77ded
function hidden_graph_view(model, graph_number)
	number_hg = length(model.layers[1][:])
	h_adj = model.layers[1][graph_number].h_adj
	h_nf = model.layers[1][graph_number].h_nf
	n_ef = model.layers[1][graph_number].n_ef
	
	nv = size(h_adj)[1] # number of vertices

	adj_reg_mx = Zygote.@ignore upper_triang(nv)|> (isa(h_adj, CuArray) ? gpu : cpu) # upper triangle matrix with diag 0

	h_adj_r = stack([l.σ.((h_adj[:,:,i].*adj_reg_mx)+(h_adj[:,:,i].*adj_reg_mx)') for i ∈ 1:n_ef]) # make each layer a square matrix, apply activation
	
	h_nf_r = l.σ.(h_nf)

	graph_feature = [i == graph_number for i ∈ 1:number_hg]
	res = Chain(model.layers[2:end]...)(graph_feature)

	return h_adj_r, h_nf_r, res
end

# ╔═╡ e5b410af-a936-45ad-86e2-5f6799b67690
function hidden_graph_view2(model, graph_number)
	number_hg = length(model.layers[1][:])
	A_2 = model.layers[1][graph_number].A_2
	X_2 = model.layers[1][graph_number].X_2
	num_edge_types = model.layers[1][graph_number].num_edge_types
	
	nv = size(A_2)[1]
		
	n_edge_types = size(A_2)[3]

	A_norm_mx = upper_triang(nv)
		
	A_2_herm = stack([relu.((A_2[:,:,i].*A_norm_mx)+(A_2[:,:,i].*A_norm_mx)') for i ∈ 1:n_edge_types])

	id = Matrix{Float32}(I, nv, nv) |> (isa(A_2, CuArray) ? gpu : cpu)  
	
	A_2_adj = stack([sum([A_2_herm[:,:,i] for i ∈ 1:n_edge_types]).+id for i ∈ 1:n_edge_types])
	
	A_2_norm = A_2_herm#./A_2_adj

	X_2_norm = relu.(X_2)#./sum([relu.(X_2)[:,i] for i ∈ 1:num_edge_types])

	graph_feature = [i == graph_number for i ∈ 1:number_hg]
	res = Chain(model.layers[2:end]...)(graph_feature)

	return A_2_norm, X_2_norm, res
end

# ╔═╡ 56404800-8f83-4a28-b693-70abbccdc193
function hidden_graph(
	h;
	col_to_label::Dict=Dict(axes(h[2], 2) .=> axes(h[2], 2)),
	slice_to_label=["$(Char('`' + i))" for i in axes(h[1], 3)],
	non_bond_idx::Int=4,
	edge_threshold::Real=0.5
)
	# get adjacency matrix by edge thresholding
	adj = (h[1][:, :, non_bond_idx] .< edge_threshold) - diagm(ones(axes(h[1], 1)))

	# generate graph topology
	g = MetaGraph(SimpleGraph(adj))

	# set node features
	nx = argmax.(eachrow(h[2]))
	for v in vertices(g)
		set_prop!(g, v, :label, col_to_label[nx[v]])
	end

	# set edge weights
	idx = [i for i in axes(h[1], 3) if i ≠ non_bond_idx]
	wts = reshape(
		[slice_to_label[x[3]] for x in argmax(h[1][:, :, 1:3]; dims=3)], 
		axes(h[1][:, :, 1])
	)
	for e in edges(g)
		set_prop!(g, e, :label, wts[src(e), dst(e)])
	end
	
	return g
end;

# ╔═╡ fb88fdc4-e5f1-4632-b0f6-72acff41def5
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

# ╔═╡ 7d785b2f-7164-4b46-8717-3a115b5b2b31
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

# ╔═╡ 3e4c206d-0fb7-4fd7-bf0f-ec2f40109f8e
function make_kgnn(graph_size,n_node_types_model,n_edge_types_model,p,n_hidden_graphs)
	model = Chain(
		Parallel(vcat, 
			[KGNNLayer(graph_size,n_node_types_model,n_edge_types_model,p) for i ∈ 1:n_hidden_graphs]...
		),device,
		Dense(n_hidden_graphs => 2, tanh),
	    #Dense(4 => 4, tanh),
		#Dropout(0.2),
		#Dense(4 => 2, tanh),
		softmax,
	)
	return model
end

# ╔═╡ e92f150b-fc55-45c5-b195-a7feb79dc413
function train_model_batch_dist(class_labels, graph_vector, test_classes, test_graphs, model, batch_sz::Int, n::Int, lr::Float64)
	#good values: n = 900, lr = .1

	n_samples = length(graph_vector)
	oh_class = Flux.onehotbatch(class_labels, [true, false])
	test_class = Flux.onehotbatch(test_classes, [true, false])
	data_class = zip(graph_vector, [oh_class[:,i] for i ∈ 1:n_samples])

	optim = Flux.setup(Flux.Adam(lr/n_samples), model)
	
	losses = []
	test_similarity = []
	test_accuracy = []

	for epoch in 1:n

		batches = Base.Iterators.partition(shuffle(collect(data_class)), 3) |> device
		epoch_loss = 0
		
		for batch ∈ batches

            results = map(batch) do (x,y)
                loss, grads = Flux.withgradient(model) do m
    
			        y_hat = m(x)
			        Flux.mse(y_hat, y)
			        
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
		
		epoch_sim = sum(pred_vector'test_class)/length(pred_bool)
		epoch_acc = sum(pred_bool.==test_classes)/length(pred_bool)
		
		push!(losses, epoch_loss/n_samples)
		push!(test_accuracy, epoch_acc)
		push!(test_similarity, epoch_sim)
	end
	return losses, model, test_accuracy, test_similarity
end

# ╔═╡ 7132824a-fdfe-4443-ad96-23bd7e188c03
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
				Flux.mse(y_hat, y)
				
			end
			push!(losses, loss)
			Flux.update!(optim, model, grads[1]|>device)
		end
        
	end
	return losses, model
end

# ╔═╡ 1a1f354d-a105-410f-b5aa-d4b38ecbacba
#train_model_graph_batch(graph_classes, btx_featuredgraphs, make_kgnn(8,12,4,4,4)|>device,1,1,.1)

# ╔═╡ 98f48ded-4129-47d5-badd-a794f09d42cb
begin #Training Data Preparation
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

end

# ╔═╡ 198336e0-c2b2-48de-8ec4-9a006719dbba
function train_kgnn(
	graph_data::Vector, 
	class_data::Vector;
	lr = 0.05,
	n_epoch = 600::Int,
	p = 4::Int,
	n_hg = 8::Int,
	size_hg = 5::Int,
	device = gpu,
	batch_sz = 1,
	train_portion = 0.8,
	test_portion = 0.2
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
	test_set_sz = Int(round(test_portion*length(featured_graphs)))

	shuffled_data = shuffle(collect(zip(featured_graphs,class_labels)))
	
	training_data = shuffled_data[1:train_set_sz]
	
	testing_data = shuffled_data[train_set_sz+1:min(train_set_sz+test_set_sz+1,size(shuffled_data)[1])]

	validation_data = []
	
	if train_set_sz+test_set_sz+1 < size(shuffled_data)[1]
		validation_data = shuffled_data[train_set_sz+test_set_sz+1:end]
	end
	
	training_graphs = getindex.(training_data,1)
	training_classes = getindex.(training_data,2)

	testing_graphs = getindex.(testing_data,1)
	testing_classes = getindex.(testing_data,2)

	losses, trained_model, epoch_test_accuracy, epoch_test_similarity = train_model_batch_dist(
		training_classes, 
		training_graphs,
		testing_classes,
		testing_graphs,
		make_kgnn(
			size_hg,
			n_node_types,
			n_edge_types,
			p,
			n_hg
		)|>device,
		batch_sz,
		n_epoch,
		lr
	)

	output_model = cpu(trained_model)

	data = (;training_data, testing_data, validation_data, labels)
	
	return (;losses, output_model, epoch_test_accuracy, data)
end

# ╔═╡ 0cb6e4a6-1acc-45ee-b25a-0f5b2c5a02e5
res = train_kgnn(btx_graphs,graph_classes, lr = 1.0, n_epoch = 300)

# ╔═╡ 3033b3a3-8f89-41d8-a8c8-fcedfdcf03ef
#@load "C:\\Users\\dcase\\RWKLayer\\res.jld2" res

# ╔═╡ e290c142-57de-477f-bd87-f2e9f95f8d2c
res

# ╔═╡ e3df7446-a75d-4d97-8eef-988910e55753
#begin
	#plot(res.losses, yaxis="Loss")
	#plot!(twinx(),res.epoch_test_accuracy, yaxis = "accuracy", linecolor = "light green")
	#plot!(twinx(),res.epoch_test_similarity, linecolor = "yellow")
#end

# ╔═╡ b9a830ee-396d-49eb-833f-685e248734c0
@save "C:\\Users\\dcase\\RWKLayer\\res.jld2" res

# ╔═╡ 54e767c0-dcd0-4825-aeb0-456e66cdc538
CUDA.reclaim()

# ╔═╡ a2d097a3-cc32-491a-9b23-420cef1c57ad
#function comtribution_map(G::MetaGraph, model; k=4)
begin
	investigated_model = res.output_model |> device
	
	G = btx_graphs[9]

	vertex_counts = Dict(zip(vertices(G), zeros(length(vertices(G)))))
	vertex_outcomes = Dict(zip(vertices(G), zeros(length(vertices(G)))))
	
	graphlets = MolecularGraphKernels.con_sub_g(4,G)
	
	graph_vec = [induced_subgraph(G,graphlet)[1] for graphlet ∈ graphlets]
	
	featuredgraph_vec = mg_to_fg(graph_vec,res.data.labels.edge_labels,res.data.labels.vertex_labels)#mg_to_fg(graph_vec,res.data.labels.edge_labels,res.data.labels.vertex_labels)
	
	model_outs = [investigated_model(featuredgraph|>device) for featuredgraph ∈ featuredgraph_vec]|>cpu

	for i ∈ eachindex(model_outs)
		model_prediction = model_outs[i]
		vertices_present = graphlets[i]
		for v ∈ vertices_present
			vertex_counts[v] +=1
			vertex_outcomes[v] += (model_prediction[1]-model_prediction[2])
			
		end
	end
	for i ∈ eachindex(vertex_outcomes)
		vertex_outcomes[i] = vertex_outcomes[i]/vertex_counts[i]
	end
end

# ╔═╡ a8696683-be0b-4909-bf97-0fb51514d252
vertex_outcomes

# ╔═╡ Cell order:
# ╠═def5bc20-8835-4484-82ca-1cee86d9a34e
# ╠═b01bcf77-efb2-41db-bd16-dd285ba090e0
# ╠═ae30e674-5e45-461a-9249-33ac79e44fd6
# ╠═522c0377-b1f5-4768-9db8-9a7bec01311a
# ╠═1bf932f8-05c6-4237-962c-9e99c5c29004
# ╠═10cdfcd0-d2e7-4ca6-a113-f944b1ddb99c
# ╠═48fb7ca7-0acc-4c42-95d4-45f22ecd3817
# ╠═4ef939f4-9e6e-4ae0-96c3-0331afcfd195
# ╠═3658ade9-e6ec-4445-8ea7-51bf51c77ded
# ╠═e5b410af-a936-45ad-86e2-5f6799b67690
# ╠═56404800-8f83-4a28-b693-70abbccdc193
# ╠═fb88fdc4-e5f1-4632-b0f6-72acff41def5
# ╠═7d785b2f-7164-4b46-8717-3a115b5b2b31
# ╠═3e4c206d-0fb7-4fd7-bf0f-ec2f40109f8e
# ╠═e92f150b-fc55-45c5-b195-a7feb79dc413
# ╠═7132824a-fdfe-4443-ad96-23bd7e188c03
# ╠═1a1f354d-a105-410f-b5aa-d4b38ecbacba
# ╠═98f48ded-4129-47d5-badd-a794f09d42cb
# ╠═198336e0-c2b2-48de-8ec4-9a006719dbba
# ╠═0cb6e4a6-1acc-45ee-b25a-0f5b2c5a02e5
# ╠═3033b3a3-8f89-41d8-a8c8-fcedfdcf03ef
# ╠═e290c142-57de-477f-bd87-f2e9f95f8d2c
# ╠═e3df7446-a75d-4d97-8eef-988910e55753
# ╠═b9a830ee-396d-49eb-833f-685e248734c0
# ╠═54e767c0-dcd0-4825-aeb0-456e66cdc538
# ╠═a2d097a3-cc32-491a-9b23-420cef1c57ad
# ╠═a8696683-be0b-4909-bf97-0fb51514d252
