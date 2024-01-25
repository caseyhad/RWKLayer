 
 

using Markdown
using InteractiveUtils

 
begin
	import Pkg
	Pkg.activate()
	using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels
	TableOfContents(title="Random walk layer w/ edge labels")
end

 
device = gpu

 
function kron2d(A, B)
    A4 = reshape(A, 1, size(A, 1), 1, size(A, 2))
    B4 = reshape(B, size(B, 1), 1, size(B, 2), 1)
    C4 = A4 .* B4
    C = reshape(C4, size(A, 1) * size(B, 1), size(A, 2) * size(B, 2))
end

 
begin
	function split_adjacency(g::MetaGraph,edge_labels)::Array{Float32}
		 
		
		nv = size(g)[1]  
		nt = length(edge_labels)  
		adj_layers = zeros(Float32,nv, nv, nt)  

		 
		for l ∈ eachindex(edge_labels)
			for edge ∈ edges(g)
				if get_prop(g, edge, :label) == edge_labels[l]
					 
					adj_layers[src(edge),dst(edge),l] = 1.0
				end
			end
			 
			adj_layers[:,:,l] = adj_layers[:,:,l]+adj_layers[:,:,l]'
		end
		
		return adj_layers
	end
	 
	function split_adjacency(fg::AbstractFeaturedGraph)::Array{Float32}
		ef = fg.ef.signal  
		nv = size(fg.nf.signal)[2]
		ne = size(ef)[2]
		nt = size(ef)[1]  
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

 
function upper_triang(sz)  
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

 
begin
	struct KGNN <: AbstractGraphLayer
	    h_adj  
		h_nf  
		n_ef   
		p  
	    σ  
	end

	 
	 
	KGNNLayer(num_nodes, num_node_types, num_edge_types, p) = KGNN(
		Float32.(rand(num_nodes, num_nodes, num_edge_types)), 
		Float32.(rand(num_nodes,num_node_types)), 
		Int(num_edge_types), 
		Int(p), 
		σ
	)
	
	@functor KGNN

	 
	function (l::KGNN)(A, X)::Float32

		nv = size(l.h_adj)[1]  

		adj_reg_mx = Zygote.@ignore upper_triang(nv)|> (isa(l.h_adj, CuArray) ? gpu : cpu)  

		h_adj_r = stack([l.σ.((l.h_adj[:,:,i].*adj_reg_mx)+(l.h_adj[:,:,i].*adj_reg_mx)') for i ∈ 1:l.n_ef])  
		
		h_nf_r = l.σ.(l.h_nf)
		
		k_xy = sum((vec(X*h_nf_r')*vec(X*h_nf_r')'.*sum([kron2d(h_adj_r[:,:,i],A[:,:,i]) for i ∈ 1:l.n_ef]))^l.p)  
		
		k_xx = sum((vec(X*X')*vec(X*X')'.*sum([kron2d(A[:,:,i],A[:,:,i]) for i ∈ 1:l.n_ef]))^l.p)  
			
		k_yy = sum((vec(h_nf_r*h_nf_r')*vec(h_nf_r*h_nf_r')'.*sum([kron2d(h_adj_r[:,:,i],h_adj_r[:,:,i]) for i ∈ 1:l.n_ef]))^l.p)  

		return k_xy/(k_xx*k_yy)^.5  
	
	end
	
	function (l::KGNN)(fg::AbstractFeaturedGraph)
	    return l(global_feature(fg), node_feature(fg)')
	end
end

 
begin  
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

 
function mg_to_fg(meta_graphs::Vector{MetaGraphs.MetaGraph{Int64, Float64}}, edge_labels::Vector, node_labels::Vector)::Array{AbstractFeaturedGraph}  
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
	
	nv = size(h_adj)[1]  

	adj_reg_mx = Zygote.@ignore upper_triang(nv)|> (isa(h_adj, CuArray) ? gpu : cpu)  

	h_adj_r = stack([l.σ.((h_adj[:,:,i].*adj_reg_mx)+(h_adj[:,:,i].*adj_reg_mx)') for i ∈ 1:n_ef])  
	
	h_nf_r = l.σ.(h_nf)

	graph_feature = [i == graph_number for i ∈ 1:number_hg]
	res = Chain(model.layers[2:end]...)(graph_feature)

	return h_adj_r, h_nf_r, res
end

 
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
	
	A_2_norm = A_2_herm 

	X_2_norm = relu.(X_2) 

	graph_feature = [i == graph_number for i ∈ 1:number_hg]
	res = Chain(model.layers[2:end]...)(graph_feature)

	return A_2_norm, X_2_norm, res
end

 
function hidden_graph(
	h;
	col_to_label::Dict=Dict(axes(h[2], 2) .=> axes(h[2], 2)),
	slice_to_label=["$(Char('`' + i))" for i in axes(h[1], 3)],
	non_bond_idx::Int=4,
	edge_threshold::Real=0.5
)
	 
	adj = (h[1][:, :, non_bond_idx] .< edge_threshold) - diagm(ones(axes(h[1], 1)))

	 
	g = MetaGraph(SimpleGraph(adj))

	 
	nx = argmax.(eachrow(h[2]))
	for v in vertices(g)
		set_prop!(g, v, :label, col_to_label[nx[v]])
	end

	 
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

 
function make_kgnn(graph_size,n_node_types_model,n_edge_types_model,p,n_hidden_graphs)
	model = Chain(
		Parallel(vcat, 
			[KGNNLayer(graph_size,n_node_types_model,n_edge_types_model,p) for i ∈ 1:n_hidden_graphs]...
		),device,
		Dense(n_hidden_graphs => 2, tanh),
	     
		 
		 
		softmax,
	)
	return model
end

 
function train_model_batch_dist(class_labels, graph_vector, test_classes, test_graphs, model, batch_sz::Int, n::Int, lr::Float64)
	 

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

 
function train_model_graph_batch(class_labels, graph_vector, model, batch_sz, n, lr,)
	 

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

	graph_classes = btx_class_labels[[i ∉[errored_smiles[i][1] for i ∈ 1:length(errored_smiles)] for i in 1:length(btx_class_labels)]]

end

 
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
	train_portion = 0.7,
	test_portion = 0.15
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
	
	testing_data = shuffled_data[train_set_sz+1:train_set_sz+test_set_sz+1]
	
	validation_data = shuffled_data[train_set_sz+test_set_sz+1:end]
	
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

	data = (;training_data, testing_data, validation_data)
	
	return (;losses, trained_model, epoch_test_accuracy, epoch_test_similarity, data)
end

#begin script execution
res = train_kgnn(btx_graphs,graph_classes, lr = 0.1, n_epoch = 1)
 
@save pwd()*"\\res.jld2" res


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
