### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ def5bc20-8835-4484-82ca-1cee86d9a34e
begin
	import Pkg
	Pkg.activate()
	using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, PlutoUI, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels, Distributed
	TableOfContents(title="Random walk layer w/ edge labels")
end

# ╔═╡ 522c0377-b1f5-4768-9db8-9a7bec01311a
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
	A_2 = model.layers[1][graph_number].A_2
	X_2 = model.layers[1][graph_number].X_2
	num_edge_types = model.layers[1][graph_number].num_edge_types
	
	nv = size(A_2)[1]
		
	n_edge_types = size(A_2)[3]

	A_norm_mx = upper_triang(nv)
		
	A_2_herm = stack([relu.((A_2[:,:,i].*A_norm_mx)+(A_2[:,:,i].*A_norm_mx)') for i ∈ 1:n_edge_types])

	id = Matrix{Float32}(I, nv, nv) |> (isa(A_2, CuArray) ? gpu : cpu)  
	
	A_2_adj = stack([sum([A_2_herm[:,:,i] for i ∈ 1:n_edge_types]).+id for i ∈ 1:n_edge_types])
	
	A_2_norm = A_2_herm./A_2_adj

	X_2_norm = relu.(X_2)./sum([relu.(X_2)[:,i] for i ∈ 1:num_edge_types])

	graph_feature = [i == graph_number for i ∈ 1:number_hg]
	res = Chain(model.layers[2:end]...)(graph_feature)

	return A_2_norm, X_2_norm, res
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

# ╔═╡ 98f48ded-4129-47d5-badd-a794f09d42cb
begin
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

end

# ╔═╡ 4d2c02fb-1a7c-435d-8a2d-f058f8442fba
graph_classes = btx_class_labels[[i ∉[errored_smiles[i][1] for i ∈ 1:length(errored_smiles)] for i in 1:length(btx_class_labels)]]

# ╔═╡ 3e4c206d-0fb7-4fd7-bf0f-ec2f40109f8e
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

# ╔═╡ c6e2f254-5bad-453e-88c6-b2c3f0f91a00
function train_model_batch_mt(class_labels,graph_vector,model, batch_sz)
	#good values: n = 900, lr = .1
	n = 4000
	lr = .1
	target = Flux.onehotbatch(class_labels, [true, false])
	loader = []
	n_samples = length(graph_vector)
	
	for i ∈ 1:n_samples
		push!(loader,(graph_vector[i],target[:,i]))
	end

	optim = Flux.setup(Flux.Adam(lr/n_samples), model)
	
	losses = []

	threads = Threads.nthreads()

	for epoch in 1:n

		randomize = randperm(length(collect(1:n_samples)))
		batches = collect(Base.Iterators.partition(randomize,batch_sz))
		
		grads_s = [Vector{Any}() for i ∈ 1:threads]
		loss_s = zeros(threads)
		
		for batch ∈ batches
			Threads.@threads for (x, y) in loader[batch]
				threadid = Threads.threadid()
				
					loss, grads = Flux.withgradient(model) do m
					
						y_hat = m(x)
			            Flux.mse(y_hat, y)
						
					end
				
				loss_s[threadid] += loss
				push!(grads_s[threadid],grads)
	
				grads_s[threadid] = [reduce(add_gradients, grads_s[threadid])]
			end
			∇ = reduce(add_gradients, reduce(vcat,grads_s))
		
			Flux.update!(optim, model, ∇[1])
		end

		push!(losses, sum(loss_s)/n_samples)
	end
	return losses, model
end

# ╔═╡ c0a5b9b4-169a-466d-ab3b-613c04d515cc


# ╔═╡ 3d2af240-b955-462a-a674-526f004c9306
function remote_calc(model,x,y)
	
		loss, grads = Flux.withgradient(model) do m
		
			y_hat = m(x)
			Flux.mse(y_hat, y)
			
		end
	return (;loss, grads)
end

# ╔═╡ 38834ae0-d289-4a5d-87b8-ce230bd0fe81
function train_model_batch_dist(class_labels,graph_vector,model, batch_sz)
	#good values: n = 900, lr = .1
	n = 4000
	lr = .1
	target = Flux.onehotbatch(class_labels, [true, false])
	loader = []
	n_samples = length(graph_vector)
	
	for i ∈ 1:n_samples
		push!(loader,(graph_vector[i],target[:,i]))
	end

	optim = Flux.setup(Flux.Adam(lr/n_samples), model)
	
	losses = []

	threads = Threads.nthreads()

	for epoch in 1:n

		randomize = randperm(length(collect(1:n_samples)))
		batches = collect(Base.Iterators.partition(randomize,batch_sz))
		
		grads_s = [Vector{Any}() for i ∈ 1:threads]
		loss_s = zeros(threads)
		
		for batch ∈ batches
			@show batch
			results = pmap(batch) do x, y
				return remote_calc(model, x, y)
			end
			
			∇ = reduce(add_gradients, reduce(vcat,grads_s))

			Flux.update!(optim, model, ∇[1])
		end

		push!(losses, sum(loss_s)/n_samples)
	end
	return losses, model
end

# ╔═╡ 939d88cd-1b00-44e6-9920-d187bdac2869
btx_losses, btx_model= train_model_batch_dist(graph_classes[1:1],btx_featuredgraphs[1:1],make_kgnn(3,12,4,4,1),1)

# ╔═╡ 067f8a07-b1b5-4e37-bd46-503ce7c0aaa2
remote_calc(make_kgnn(3,12,4,4,1),btx_featuredgraphs[1],graph_classes[1])

# ╔═╡ 6b1502fd-8574-4611-8a6e-cff63c0c327f


# ╔═╡ f342e320-3bc0-4a5d-9dc0-1479b827e9f3
plot(btx_losses)

# ╔═╡ 1794f38a-ba9c-4332-8710-a683addd18c1
btx_model.layers[1][1]

# ╔═╡ 75c25f96-8fe8-4c65-a3c6-59e58f71152a
hidden_graph_view2(btx_model, 5)

# ╔═╡ 3f07d95a-bccb-405d-b062-f361af3fc31d
begin
	test_prediction = [1-(maximum(btx_model(btx_featuredgraphs[i])).==btx_model(btx_featuredgraphs[i])[2]) for i ∈ 1:90]

	test = btx_class_labels[1:90]

	tp = sum([test_prediction[i]==1&&test[i]==1 for i ∈ 1:length(test)])

	tn = sum([test_prediction[i]==0&&test[i]==0 for i ∈ 1:length(test)])

	fp = sum([test_prediction[i]==1&&test[i]==0 for i ∈ 1:length(test)])

	fn = sum([test_prediction[i]==0&&test[i]==1 for i ∈ 1:length(test)])
	
	pre = tp/(tp+fp)

	rec = tp/(tp+fn)

	acc = mean(test_prediction.==test)

	f1 = 2*tp/(2*tp+fp+fn)

end

# ╔═╡ ba6c41b4-a1b6-49ba-894a-35f0495d44a0
f1

# ╔═╡ 12ba95cd-f221-4d2a-a022-ad231188bfb4
btx_featuredgraphs

# ╔═╡ 869d338e-59c7-404c-a49f-7f6793d813d4
btx_class_labels

# ╔═╡ Cell order:
# ╠═def5bc20-8835-4484-82ca-1cee86d9a34e
# ╠═522c0377-b1f5-4768-9db8-9a7bec01311a
# ╠═1bf932f8-05c6-4237-962c-9e99c5c29004
# ╠═10cdfcd0-d2e7-4ca6-a113-f944b1ddb99c
# ╠═48fb7ca7-0acc-4c42-95d4-45f22ecd3817
# ╠═4ef939f4-9e6e-4ae0-96c3-0331afcfd195
# ╠═3658ade9-e6ec-4445-8ea7-51bf51c77ded
# ╠═e5b410af-a936-45ad-86e2-5f6799b67690
# ╠═56404800-8f83-4a28-b693-70abbccdc193
# ╠═fb88fdc4-e5f1-4632-b0f6-72acff41def5
# ╠═98f48ded-4129-47d5-badd-a794f09d42cb
# ╠═4d2c02fb-1a7c-435d-8a2d-f058f8442fba
# ╠═3e4c206d-0fb7-4fd7-bf0f-ec2f40109f8e
# ╠═c6e2f254-5bad-453e-88c6-b2c3f0f91a00
# ╠═38834ae0-d289-4a5d-87b8-ce230bd0fe81
# ╠═c0a5b9b4-169a-466d-ab3b-613c04d515cc
# ╠═3d2af240-b955-462a-a674-526f004c9306
# ╠═939d88cd-1b00-44e6-9920-d187bdac2869
# ╠═067f8a07-b1b5-4e37-bd46-503ce7c0aaa2
# ╠═6b1502fd-8574-4611-8a6e-cff63c0c327f
# ╠═6febe4b5-1770-42f9-98a0-95befa84508d
# ╠═f342e320-3bc0-4a5d-9dc0-1479b827e9f3
# ╠═1794f38a-ba9c-4332-8710-a683addd18c1
# ╠═75c25f96-8fe8-4c65-a3c6-59e58f71152a
# ╠═3f07d95a-bccb-405d-b062-f361af3fc31d
# ╠═ba6c41b4-a1b6-49ba-894a-35f0495d44a0
# ╠═12ba95cd-f221-4d2a-a022-ad231188bfb4
# ╠═869d338e-59c7-404c-a49f-7f6793d813d4
