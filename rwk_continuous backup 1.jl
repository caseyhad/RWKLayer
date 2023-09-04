### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ a0936f30-ff65-11ed-352c-c795c803f250
begin
	import Pkg
	Pkg.activate()
	using MolecularGraph, MolecularGraphKernels, Test, MetaGraphs, Graphs, ProfileCanvas, PlutoUI, BenchmarkTools, Flux, Functors, Zygote, LinearAlgebra, Plots, CUDA, JLD2, GeometricFlux, Kronecker
end

# ╔═╡ 643fcfc0-f38b-4f18-954d-4355e840b032
device = cpu

# ╔═╡ b1a5c617-9543-4294-a557-f735d5ac7365
CUDA.versioninfo()

# ╔═╡ 7ea77cb0-7037-4a24-9799-1e94fb2219aa
begin # Demonstration that the Kronecker based approach and MGK have the same answer for K(x,y) between two molecular graphs
	g₁ = MetaGraph(removehydrogens(smilestomol("C[C@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@@H](O1)O[C@H](C2=O)[C@H](c(c3)ccc(O)c3O)Oc(c24)cc(O)cc4O")))
	g₂ = MetaGraph(removehydrogens(smilestomol("O[C@@H](O1)[C@H](O)[C@@H](O)[C@H](O)[C@H]1CO[C@@H](O2)[C@H](O)[C@@H](O)[C@@H](O)[C@H]2CO")))
	dpg = ProductGraph{Direct}(g₁, g₂)
end

# ╔═╡ 97888b1f-36bc-4731-b261-b863bb7e7e03
begin
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
end

# ╔═╡ 433d45a2-e8be-44a4-b993-b4852c46546d
begin # preparing a small size 4 FeaturedGraph for testing the KGNN layer
	adjm = [0 1 0 1;
            1 0 1 1;
            0 1 0 1;
            1 1 1 0];

	nf = [1 0 0 1
	 	  0 1 0 0
	      0 0 1 0]
	nf_norm = (nf'./sum.(eachrow(nf')))'
	ef = [1 0 0 1 1 
	 	  0 1 0 0 0 
	 	  0 0 1 0 0 ]
	ef_norm = (ef'./sum.(eachrow(ef')))'
	
	fg = FeaturedGraph(Float32.(adjm); nf=Float32.(nf_norm), ef=Float32.(ef_norm))

 	Ã = split_adjacency(fg)

	fg = FeaturedGraph(adjm; nf=fg.nf, ef=fg.ef, gf = Ã) |> device

end

# ╔═╡ 376b18e1-8b9f-4103-a5c9-78e1ea85c082
function rwk_kron(g₁::MetaGraph,g₂::MetaGraph;l=1)
	# The non-normalized kernel calculaion for two MetaGraphs
	
	edge_labels = unique(
		[get_prop(g₁, edge, :label) for edge ∈ edges(g₁)]
		∪
		[get_prop(g₂, edge, :label) for edge ∈ edges(g₂)]
	)
	
	vertex_labels = unique(
		[get_prop(g₁, v, :label) for v ∈ vertices(g₁)]
		∪
		[get_prop(g₂, v, :label) for v ∈ vertices(g₂)]
	)
	
	A₁ = split_adjacency(g₁,edge_labels)
	A₂ = split_adjacency(g₂,edge_labels)
	
	X₁ = zeros(size(g₁)[1],length(vertex_labels))
	X₂ = zeros(size(g₂)[1],length(vertex_labels))
	
	for i ∈ eachindex(vertices(g₁))
		X₁[i,:] = vertex_labels.==get_prop(g₁, i, :label)
	end
	
	for i ∈ eachindex(vertices(g₂))
		X₂[i,:] = vertex_labels.==get_prop(g₂, i, :label)
	end

	# formula for the kronecker-based random walk kernel of graphs with node and edge labels
	return sum((vec(X₁*X₂')*vec(X₁*X₂')'.*sum([kron(A₂[:,:,i],A₁[:,:,i]) for i ∈ eachindex(edge_labels)]))^l)
	
end
	

# ╔═╡ 2f3098ae-a36e-4d90-8241-32fe70259cb6
rwk_kron(g₁,g₂;l=5)

# ╔═╡ 3370544e-76fb-4b15-b59a-1cae00d2e514
MolecularGraphKernels.random_walk(ProductGraph{Direct}(g₁, g₂); l=5)

# ╔═╡ d6df0686-dbe2-4c4f-a795-f90c24c35cb2
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

# ╔═╡ 2a77361d-4872-4aa2-b63a-2d367bd16ec4
begin
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
		A_2_herm = stack([relu.((l.A_2[:,:,i].*A_norm_mx)+(l.A_2[:,:,i].*A_norm_mx)') for i ∈ 1:4])

		id = Matrix{Float32}(I, nv, nv) |> (isa(l.A_2, CuArray) ? gpu : cpu) # identity matrix on cpu or fpu depending on which is being used
		
		A_2_adj = stack([sum([A_2_herm[:,:,i] for i ∈ 1:n_edge_types]).+id for i ∈ 1:n_edge_types]) # a slice of this matrix houses the sum of an edge label vector for each edge. the identity matrix is added to avoid dividing by 0 on the main daigonal. The matrix is copied for each slice to match dimensions of A_2_herm
		
		A_2_norm = A_2_herm./A_2_adj # result - all edge feature vectors sum to 1

		# same result as edge features except for node features
		X_2_norm = relu.(l.X_2)./sum([relu.(l.X_2)[:,i] for i ∈ 1:l.num_edge_types])
		

		# inner product normalization - k(x,y)/(k(x,x)*k(y,y))^.5
		return sum((vec(X*X_2_norm')*vec(X*X_2_norm')'.*sum([kron(A_2_norm[:,:,i],A[:,:,i]) for i ∈ 1:l.num_edge_types]))^l.p)/(sum((vec(X*X')*vec(X*X')'.*sum([kron(A[:,:,i],A[:,:,i]) for i ∈ 1:l.num_edge_types]))^l.p)*sum((vec(X_2_norm*X_2_norm')*vec(X_2_norm*X_2_norm')'.*sum([kron(A_2_norm[:,:,i],A_2_norm[:,:,i]) for i ∈ 1:l.num_edge_types]))^l.p))^.5
	
	end
	
	function (l::KGNN)(fg::AbstractFeaturedGraph)
	    return l(global_feature(fg), node_feature(fg)')
	end
end

# ╔═╡ 863ae18c-eb5e-45ab-ac0a-6ba0363a5603
begin # isomorphism learning test with one size 4 hidden graph, loss is 1-kernel score
	A_2 = Float32.(rand(4,4, 4))
	model = KGNN(A_2, Float32.(rand(4,3)), 3, 4, relu) |> device
	optim = Flux.setup(Flux.Momentum(1, 0.5), model)
	losses = []
	loss = []
	for epoch in 1:300
		loss, grads = Flux.withgradient(model) do m
			ker_norm = m(fg)
			1-ker_norm # loss function
		end
		Flux.update!(optim, model, grads[1]) # update model using the gradient
		push!(losses, loss)
	end
end


# ╔═╡ 35ed4380-8cf0-4193-9c9a-d497f0af344d
plot(losses)

# ╔═╡ 5f8872b1-838f-4c21-814c-f0eaee9da8ac
mcpu = (model |> cpu)

# ╔═╡ 1107ffd6-347e-4c4b-bd56-e35cc445e8e1
mcpu(fg |> cpu)

# ╔═╡ 319b6b2f-0b17-4123-8c54-895db1ec2bf7
begin # same logic as inside the kernel layer, to replicate the hidden graph from the "point of view" of the kernel function
	nv = size(mcpu.A_2)[1]
		
	n_edge_types = size(mcpu.A_2)[3]

	A_norm_mx = upper_triang(nv)
		
	A_2_herm = stack([relu.((mcpu.A_2[:,:,i].*A_norm_mx+(mcpu.A_2[:,:,i].*A_norm_mx)')) for i ∈ 1:4])

	id = Matrix{Float32}(I, nv, nv) |> (isa(mcpu.A_2, CuArray) ? gpu : cpu)  
	
	A_2_adj = stack([sum([A_2_herm[:,:,i] for i ∈ 1:n_edge_types]).+id for i ∈ 1:n_edge_types])
	
	A_2_norm = A_2_herm./A_2_adj
end

# ╔═╡ fd430464-1093-47b7-be83-d0fe14a87547
relu.(mcpu.X_2)./sum([relu.(mcpu.X_2)[:,i] for i ∈ 1:mcpu.num_edge_types])

# ╔═╡ 000a1031-fbfe-4891-a565-64d627a37a6d
split_adjacency(fg |> cpu)

# ╔═╡ 8294c34a-f1a2-4571-8199-e78a41c22825
fg.nf

# ╔═╡ 4348d758-73cf-4575-819c-4759acb1622f
begin # load synthetic test set #1
	synthetic_data = load(string(pwd(),"\\molecules.jld2"))
	synthetic_graph_data = synthetic_data["molecules"]
	synthetic_classes = synthetic_data["class_enc"]
end

# ╔═╡ 6a56f67b-9318-4fbf-a448-221e0cd6e522
viz_graph(synthetic_graph_data[25])

# ╔═╡ 324da530-d3c1-4c00-82b9-6f27da5e56f2
graph_labels = [get_prop(g, :label) for g in synthetic_graph_data]

# ╔═╡ 4b88d981-7326-4bac-954d-5b299a475e3d
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
		

# ╔═╡ 9fe8d0dc-f431-4e69-b925-2fe01a63bbf9
viz_graph(synthetic_graph_data[1])

# ╔═╡ ffaebf5b-03e6-4ac4-b037-ee1caa74eeca
synthetic_feature_graphs = mg_to_fg(synthetic_graph_data,[-1,1,2],[6,7,8])

# ╔═╡ a2dacd2a-acba-46fa-8919-70302c6541ac
synthetic_feature_graphs[1]

# ╔═╡ ba1c5e76-7d7b-4568-b03f-d317f98df70d
begin
	target = Flux.onehotbatch(graph_labels, [true, false])
	loader = []
	for i ∈ 1:length(synthetic_feature_graphs)
		push!(loader,(synthetic_feature_graphs[i],target[:,i]))
	end
	graph_size = 4
	n_node_types_model = 3
	n_edge_types_model = 3
	p = 3
	n_hidden_graphs = 4
	model_synth = Chain(
		Parallel(vcat, 
			[KGNNLayer(graph_size,n_node_types_model,n_edge_types_model,p) for i ∈ 1:n_hidden_graphs]...
		),
		Dense(n_hidden_graphs => 4, tanh),
	    Dense(4 => 4, tanh),
		#Dropout(0.2),
		Dense(4 => 2, tanh),
		softmax,
	) |> device
	
	optim_synth = Flux.setup(Flux.Adam(.001), model_synth)
	losses_s = []
	loss_s = []
	epoch_loss = []
	for epoch in 1:300
		for (x, y) in loader[5:40]
			loss_s, grads_synth = Flux.withgradient(model_synth) do m
				
				y_hat = m(x)
	            Flux.crossentropy(y_hat, y)
				
			end
			Flux.update!(optim_synth, model_synth, grads_synth[1])
			push!(epoch_loss, loss_s)
		end
		push!(losses_s, mean(epoch_loss))
		epoch_loss = []
	end
end

# ╔═╡ 9c71c66f-907d-4c84-89b4-0a38ece28c62
plot(losses_s)

# ╔═╡ 342cf369-2143-401b-a6e2-73a1ff89710f
model_predictions = stack([maximum(model_synth(graph)) .== model_synth(graph) for graph ∈ synthetic_feature_graphs])

# ╔═╡ 264a61cf-e204-4656-bc6b-0dac3c7c98c2
sum([model_predictions[:,i] == target[:,i] for i ∈ 40:length(model_predictions[1,:])])/length(40:length(model_predictions[1,:]))

# ╔═╡ f5ce99ea-6684-4701-9dec-e0922d013a08
function hidden_graph_view(model, graph_number,number_hg)
	A_2 = model.layers[1][graph_number].A_2
	X_2 = model.layers[1][graph_number].X_2
	num_edge_types = model.layers[1][graph_number].num_edge_types
	
	nv = size(A_2)[1]
		
	n_edge_types = size(A_2)[3]

	A_norm_mx = upper_triang(nv)
		
	A_2_herm = stack([relu.((A_2[:,:,i].*A_norm_mx)+(A_2[:,:,i].*A_norm_mx)') for i ∈ 1:4])

	id = Matrix{Float32}(I, nv, nv) |> (isa(A_2, CuArray) ? gpu : cpu)  
	
	A_2_adj = stack([sum([A_2_herm[:,:,i] for i ∈ 1:n_edge_types]).+id for i ∈ 1:n_edge_types])
	
	A_2_norm = A_2_herm./A_2_adj

	X_2_norm = relu.(X_2)./sum([relu.(X_2)[:,i] for i ∈ 1:num_edge_types])

	graph_feature = [i == graph_number for i ∈ 1:number_hg]
	res = Chain(model.layers[2:end]...)(graph_feature)

	return A_2_norm, X_2_norm, res
end

# ╔═╡ 72f6259d-3b25-47a5-9166-493ef2967c75
hidden_graph_view(model_synth, 4,4)

# ╔═╡ 3fd91b06-a9fa-4b68-8b10-41ef539b148d
begin
	synthetic_data_med = load(string(pwd(),"\\molecules_med.jld2"))
	synthetic_graph_data_med = synthetic_data_med["molecules"]
	synthetic_classes_med = synthetic_data_med["class_enc"]
end

# ╔═╡ 9d71fb11-2814-45f2-b388-7f282a71b185
viz_graph(synthetic_graph_data_med[4])

# ╔═╡ 93d1b361-6da5-4375-83fb-51288097239b
graph_labels_med = [get_prop(g, :label) for g in synthetic_graph_data_med]

# ╔═╡ fdfcbe58-49d6-46b9-a23e-5ea95649934f
sum(graph_labels_med)/length(graph_labels_med)

# ╔═╡ 1e162154-2b3b-4d63-bebe-aba082421394
synthetic_feature_graphs_med = mg_to_fg(synthetic_graph_data_med,[-1,1,2],[6,7,8])

# ╔═╡ 46c4811e-7ff3-4c8b-b971-98c0b896b289
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

# ╔═╡ 24ce2183-23c0-4c79-ab72-fbe9b2b70eee
function train_model(class_labels,graph_vector,model)
	target = Flux.onehotbatch(class_labels, [true, false])
	loader = []
	
	for i ∈ 1:length(graph_vector)
		push!(loader,(graph_vector[i],target[:,i]))
	end

	model |> device
	
	optim = Flux.setup(Flux.Adam(.005), model)
	losses = []
	loss = []
	epoch_loss = []
	past_models = []
	for epoch in 1:750
		for (x, y) in loader
			loss, grads = Flux.withgradient(model) do m
				
				y_hat = m(x)
	            Flux.crossentropy(y_hat, y)
				
			end
			Flux.update!(optim, model, grads[1])
			push!(epoch_loss, loss)
		end
		push!(losses, mean(epoch_loss))
		epoch_loss = []
		push!(past_models,model)
	end
	return losses, model, past_models
end

# ╔═╡ c03e26ef-f56f-4022-9197-e8c4d57c5d2d
epoch_losses, res_model, past_models = train_model(graph_labels_med[1:90],synthetic_feature_graphs_med[1:90],make_kgnn(5,3,3,4,8))

# ╔═╡ b7ffd3e2-d3c9-4817-9d6d-ae636dc9b692
plot(epoch_losses)

# ╔═╡ de7c1f09-b8d1-4741-a163-1fcc22f2f017
hidden_graph_view(res_model, 8,8)

# ╔═╡ b5580104-8a87-4115-91bb-b3d75148aaf9
test_prediction = [1-(maximum(res_model(synthetic_feature_graphs_med[i])).==res_model(synthetic_feature_graphs_med[i])[2]) for i ∈ 90:length(synthetic_feature_graphs_med)]

# ╔═╡ 15d6b88e-75e1-46ae-b6da-995d803e1599
test = graph_labels_med[90:end]

# ╔═╡ f346c057-5416-44d7-854d-e3da68626222
(test_prediction'*test)/length(test)

# ╔═╡ bc470972-3e0c-46a0-b1ea-4b799968453e


# ╔═╡ Cell order:
# ╠═a0936f30-ff65-11ed-352c-c795c803f250
# ╠═643fcfc0-f38b-4f18-954d-4355e840b032
# ╠═128f77d4-96e4-4442-bd02-338e04ae0ad8
# ╠═b1a5c617-9543-4294-a557-f735d5ac7365
# ╠═7ea77cb0-7037-4a24-9799-1e94fb2219aa
# ╠═97888b1f-36bc-4731-b261-b863bb7e7e03
# ╠═433d45a2-e8be-44a4-b993-b4852c46546d
# ╠═376b18e1-8b9f-4103-a5c9-78e1ea85c082
# ╠═2f3098ae-a36e-4d90-8241-32fe70259cb6
# ╠═3370544e-76fb-4b15-b59a-1cae00d2e514
# ╠═2a77361d-4872-4aa2-b63a-2d367bd16ec4
# ╠═aaa6cd0f-b37d-4228-82ad-df39e0ac17bd
# ╠═d6df0686-dbe2-4c4f-a795-f90c24c35cb2
# ╠═84e3bd33-cdc4-43cf-8e93-953c01184d64
# ╠═863ae18c-eb5e-45ab-ac0a-6ba0363a5603
# ╠═35ed4380-8cf0-4193-9c9a-d497f0af344d
# ╠═5f8872b1-838f-4c21-814c-f0eaee9da8ac
# ╠═1107ffd6-347e-4c4b-bd56-e35cc445e8e1
# ╠═319b6b2f-0b17-4123-8c54-895db1ec2bf7
# ╠═fd430464-1093-47b7-be83-d0fe14a87547
# ╟─000a1031-fbfe-4891-a565-64d627a37a6d
# ╟─8294c34a-f1a2-4571-8199-e78a41c22825
# ╠═4348d758-73cf-4575-819c-4759acb1622f
# ╠═6a56f67b-9318-4fbf-a448-221e0cd6e522
# ╠═a2dacd2a-acba-46fa-8919-70302c6541ac
# ╠═324da530-d3c1-4c00-82b9-6f27da5e56f2
# ╠═4b88d981-7326-4bac-954d-5b299a475e3d
# ╠═9fe8d0dc-f431-4e69-b925-2fe01a63bbf9
# ╠═ffaebf5b-03e6-4ac4-b037-ee1caa74eeca
# ╠═ba1c5e76-7d7b-4568-b03f-d317f98df70d
# ╠═9c71c66f-907d-4c84-89b4-0a38ece28c62
# ╠═342cf369-2143-401b-a6e2-73a1ff89710f
# ╠═264a61cf-e204-4656-bc6b-0dac3c7c98c2
# ╠═f5ce99ea-6684-4701-9dec-e0922d013a08
# ╠═72f6259d-3b25-47a5-9166-493ef2967c75
# ╠═3fd91b06-a9fa-4b68-8b10-41ef539b148d
# ╠═9d71fb11-2814-45f2-b388-7f282a71b185
# ╠═93d1b361-6da5-4375-83fb-51288097239b
# ╠═fdfcbe58-49d6-46b9-a23e-5ea95649934f
# ╠═1e162154-2b3b-4d63-bebe-aba082421394
# ╠═46c4811e-7ff3-4c8b-b971-98c0b896b289
# ╠═24ce2183-23c0-4c79-ab72-fbe9b2b70eee
# ╠═c03e26ef-f56f-4022-9197-e8c4d57c5d2d
# ╠═b7ffd3e2-d3c9-4817-9d6d-ae636dc9b692
# ╠═de7c1f09-b8d1-4741-a163-1fcc22f2f017
# ╠═b5580104-8a87-4115-91bb-b3d75148aaf9
# ╠═15d6b88e-75e1-46ae-b6da-995d803e1599
# ╠═f346c057-5416-44d7-854d-e3da68626222
# ╠═bc470972-3e0c-46a0-b1ea-4b799968453e
