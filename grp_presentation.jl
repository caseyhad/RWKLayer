### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 31cf1fe0-d5a2-11ee-0215-07266138b2b6
begin
	import Pkg
	Pkg.activate(".");
	using MetaGraphs, CSV, DataFrames,MolecularGraph, JLD2, MolecularGraphKernels, GeometricFlux, Flux, CUDA, RWKLayerFunctions, Graphs, Plots, PlutoUI, LinearAlgebra, MLUtils
end

# ╔═╡ 1ea5cf04-f5e2-488e-882a-18554a3f632f
using Markdown

# ╔═╡ 98d0be39-5f65-4a7b-82fd-d99d3869b926
TableOfContents()

# ╔═╡ 59989cc7-7e6f-4330-a926-4cad0a908115
device = gpu

# ╔═╡ 0045ba50-90e4-4903-8621-26325bed6c5d
md"# **Random Walk Layer**"

# ╔═╡ 184acc05-69c2-4828-b452-f963d5391e13
Main.RWKLayerFunctions = RWKLayerFunctions

# ╔═╡ 322e22d3-ac2d-4ea1-b043-5bb9ffc0fc84
md"
The random walk kernel counts the shared walks of size k between two graphs.

A walk on the DPG maps two two identical walks in the input graphs. The kernel is typically computed by enumerating all walks on the direct product graph.

Definition of the direct product graph:
	
	Vₓ = {(vᵢ, vⱼ): vᵢ ∈ Vᵢ ∧ vⱼ ∈ Vⱼ ∧ ℓ(vᵢ) = ℓ(vⱼ)}

	
    Eₓ = {{(vᵢ, vⱼ), (uᵢ, uⱼ)} : {vᵢ, uᵢ} ∈ Eᵢ ∧ {vⱼ, uⱼ} ∈ Eⱼ ...
												∧ ℓ({vᵢ, uᵢ}) = ℓ({vⱼ, uⱼ})}

By enumerating walks using the DPG, we avoid explicit mapping to the feature vector containing counts of all walk-label sequences. Skipping this step constitutes the kernel trick.

"

# ╔═╡ 507043b5-5162-4608-b5a1-65e2cb0f2b3c
md"Graph 1:"

# ╔═╡ 1eb98b42-f8cc-4717-b725-e6dfba1d9928
md"Graph 2:"

# ╔═╡ 9a935611-1efb-48f9-aa18-c8453430d09d
md"Product Graph"

# ╔═╡ 66f23fa4-3eaf-43e9-930d-99e8e928d74f
md"
While there are faster algorithms to construct the direct product graph for two graphs, this method involves the use of the [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) of matrices to calculate a product graph that **allows continuous node/edge labels and node/edge weights** to allow the kernel algorithm to be **differentiable with respect to the node and edge weights**.

This enables us to use a graph reperesentation within a neural network, using gradient descent to _learn_ continuous graphs that make for insightful kernel comparisons with the input data.

[here](https://proceedings.neurips.cc/paper/2020/file/ba95d78a7c942571185308775a97a3a0-Paper.pdf) is an example of a proposed network design using this algorithm"



# ╔═╡ 5990adbd-d3b8-4c03-9f3e-4bfe2b86e1ab
begin # Demonstration that the Kronecker based approach and MGK have the same answer for K(x,y) between two molecular graphs
	g₁ = MetaGraph(removehydrogens(smilestomol("C[C@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@@H](O1)O[C@H](C2=O)[C@H](c(c3)ccc(O)c3O)Oc(c24)cc(O)cc4O")))
	g₂ = MetaGraph(removehydrogens(smilestomol("O[C@@H](O1)[C@H](O)[C@@H](O)[C@H](O)[C@H]1CO[C@@H](O2)[C@H](O)[C@@H](O)[C@@H](O)[C@H]2CO")))
	dpg = ProductGraph{Direct}(g₁, g₂)
end

# ╔═╡ 5e6915a9-6c0a-4f0c-bd90-d397b9e5ad12
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
	
	A₁ = RWKLayerFunctions.split_adjacency(g₁,edge_labels)
	A₂ = RWKLayerFunctions.split_adjacency(g₂,edge_labels)
	
	X₁ = zeros(size(g₁)[1],length(vertex_labels))
	X₂ = zeros(size(g₂)[1],length(vertex_labels))
	
	for i ∈ eachindex(vertices(g₁))
		X₁[i,:] = vertex_labels.==get_prop(g₁, i, :label)
	end
	
	for i ∈ eachindex(vertices(g₂))
		X₂[i,:] = vertex_labels.==get_prop(g₂, i, :label)
	end

	# formula for the kronecker-based random walk kernel of graphs with node and edge labels
	sum((vec(X₁*X₂')*vec(X₁*X₂')'.*sum([kron(A₂[:,:,i],A₁[:,:,i]) for i ∈ eachindex(edge_labels)]))^l)
	
end

# ╔═╡ bcfbc20c-4587-4efa-8173-f032ac9e609a
rwk_kron(g₁,g₂;l=5)

# ╔═╡ dc5e40f8-bd39-4769-a524-32144b4fc230
MolecularGraphKernels.random_walk(ProductGraph{Direct}(g₁, g₂); l=5)

# ╔═╡ 49b0c3b8-3b9b-47c0-b640-a0c378b1699b
begin # preparing a small size 4 FeaturedGraph for testing the KGNN layer
	adjm = [0 1 0 1;
            1 0 1 1;
            0 1 0 1;
            1 1 1 0];

	nf = [1 0 0 1
	 	  0 1 0 0
	      0 0 1 0]
	
	ef = [1 0 0 1 1 
	 	  0 1 0 0 0 
	 	  0 0 1 0 0 ]
	
	
	fg = FeaturedGraph(Float32.(adjm); nf=Float32.(nf), ef=Float32.(ef))

 	Ã = RWKLayerFunctions.split_adjacency(fg)

	fg = FeaturedGraph(adjm; nf=fg.nf, ef=fg.ef, gf = Ã) |> device

end

# ╔═╡ 62bd0181-6e87-457e-b8ef-cbd2403cdb58
html"""
<div><iframe src="https://drive.google.com/file/d/1YnV8Kt5o8CDm3lA5sDfO4ilslZDLNNto/preview" width="640" height="480" allow="autoplay"></iframe></div>
"""

# ╔═╡ a9655c63-8039-4198-8465-7c80475d93b9
@bind epoch Slider(1:2:300)

# ╔═╡ 2b42e1ae-4b73-4ee4-8318-690ebf0d11fa
function graph_isomorphism_viz(model)
	mcpu = (model |> cpu)

	nv = size(mcpu.h_adj)[1]

	n_edge_types = size(mcpu.h_adj)[3]

    adj_reg_mx = RWKLayerFunctions.upper_triang(nv)

	id = Matrix{Float32}(I, nv, nv)

    h_adj_sqr = stack([(mcpu.h_adj[:,:,i].*adj_reg_mx)+(mcpu.h_adj[:,:,i].*adj_reg_mx)'.+id for i ∈ 1:size(mcpu.h_adj)[3]])
    
	h_adj_r = permutedims(softmax(permutedims(h_adj_sqr ,(3,2,1))),(3,2,1))

    h_nf_r = stack(softmax.(eachrow(mcpu.h_nf)))'

	col_to_label = Dict(zip(1:3, 7:9))

	slice_to_label = Dict(zip(1:3, 1:3))
	
	X_2_norm = relu.(mcpu.h_nf)
	
	# get adjacency matrix by edge thresholding
	A = sum([h_adj_r[:,:,i].>0 for i in axes(h_adj_r[:,:,1:end-1], 3)])

	v_props = []
	for r in 1:size(h_adj_r)[1]

		push!(v_props,col_to_label[argmax(h_nf_r[r,:])])

	end

	# generate graph topology
	mg = MetaGraph(SimpleGraph(A))

	for v in vertices(mg)
		set_prop!(mg, v, :label, v_props[v])
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
	for e in edges(mg)
		set_prop!(mg, e, :label, wts[src(e), dst(e)])
		set_prop!(mg, e, :alpha, alphas[src(e), dst(e)])
	end

	edge_alpha_mg = [get_prop(mg,e, :alpha) for e in edges(mg)]
	return (;mg, edge_alpha_mg)
end

# ╔═╡ 74327d26-b4ae-4582-a1b6-2797ed1a0360
begin # isomorphism learning test with one size 4 hidden graph, loss is 1-kernel score
	A_2 = Float32.(rand(4,4, 4))
	model = RWKLayerFunctions.KGNN(A_2, Float32.(rand(4,3)), 3, 4, relu) |> device
	optim = Flux.setup(Flux.Momentum(1, 0.5), model)
	losses = []
	loss = []
	graph_evolution = []
	for epoch in 1:300
		loss, grads = Flux.withgradient(model) do m
			ker_norm = m(fg)
			1-ker_norm # loss function
		end
		Flux.update!(optim, model, grads[1]) # update model using the gradient
		push!(losses, loss)
		push!(graph_evolution, graph_isomorphism_viz(model))
	end
end

# ╔═╡ dc37ad80-9e1a-488e-9f9d-6a92e7166d90
RWKLayerFunctions.KGNN(A_2, Float32.(rand(4,3)), 3, 4, relu)

# ╔═╡ b5d13c10-e841-459c-b189-9e6317f8acaf
begin
	plot(-(losses).+1, linewidth=3)
	ylabel!("Kernel Score (Cosine Norm)")
	xlabel!("Epoch")
	plot!([epoch], seriestype="vline")
end

# ╔═╡ e6f246ab-7d58-4f4e-9e54-134f3d9ea80e
begin
	mg, edge_alpha = graph_evolution[epoch]
	viz_graph(mg, edge_alpha_mask=edge_alpha, layout_style = :circular)
end

# ╔═╡ fd87e5c5-2022-4d06-a9c5-955306eaac2d
begin # load synthetic test set #1
	synthetic_data = load(string(pwd(),"\\molecules.jld2"))
	synthetic_graph_data = synthetic_data["molecules"]
	synthetic_classes = synthetic_data["class_enc"]
end

# ╔═╡ 4a8339bc-18b0-46c5-8144-5a2c9a807897
viz_graph(synthetic_graph_data[15])

# ╔═╡ 4d526687-d354-4176-bbe3-20bcf27f2598
viz_graph(synthetic_graph_data[5])

# ╔═╡ 48b61aea-6a25-4143-b1be-04b839d43929
viz_graph(ProductGraph{Direct}(synthetic_graph_data[15], synthetic_graph_data[5]),layout_style = :circular)

# ╔═╡ e50bc67b-4a7c-4084-80e6-d177f15ff967
viz_graph(synthetic_graph_data[1])

# ╔═╡ f732df66-75d5-49b7-82d6-54c0f193c369
viz_graph(synthetic_graph_data[27])

# ╔═╡ 2ae2ec42-f481-4d44-a113-4ca40f1cddc1
@load pwd()*"\\results\\ne30NSVGD4.jld2" res

# ╔═╡ ee98aa2a-bdb5-4b97-bd6e-d8ba58637077
begin
	plot(res.losses, linewidth=3)
	ylabel!("Crossentropy Loss")
	xlabel!("Epoch")
	
end

# ╔═╡ 2c6d835b-ba00-4fbc-803b-00d14383e990
md"
	train_kgnn(synthetic_graph_data[5:40],graph_labels[5:40], lr = .001, n_epoch = 300, n_hg = 4, batch_sz = 1, p=3, size_hg = 4, train_portion = .95)"

# ╔═╡ 34ea3bda-2005-4104-85db-61014d634b84
begin
	hg1 = hg_to_mg(res,1)
	edge_alpha_hg1 = [get_prop(hg1,e, :alpha) for e in edges(hg1)]
	viz_graph(hg1, edge_alpha_mask=edge_alpha_hg1)
end

# ╔═╡ 24e4f3dd-5399-411a-bdcb-f17d6fff4947
begin
	hg2 = hg_to_mg(res,2)
	edge_alpha_hg2 = [get_prop(hg2,e, :alpha) for e in edges(hg2)]
	viz_graph(hg2, edge_alpha_mask=edge_alpha_hg2)
end

# ╔═╡ 58a65a8d-24f9-42e9-8794-b53144d4cae4
begin
	hg3 = hg_to_mg(res,3)
	edge_alpha_hg3 = [get_prop(hg3,e, :alpha) for e in edges(hg3)]
	viz_graph(hg3, edge_alpha_mask=edge_alpha_hg3)
end

# ╔═╡ 93e2dedd-e8a0-443e-b691-326a749ebf32
begin
	hg4 = hg_to_mg(res,4)
	edge_alpha_hg4 = [get_prop(hg4,e, :alpha) for e in edges(hg4)]
	viz_graph(hg4, edge_alpha_mask=edge_alpha_hg4)
end

# ╔═╡ c40bcbd8-85b6-4f00-b43d-c0db20c003ec
G = synthetic_graph_data[27]

# ╔═╡ b153936e-e321-454a-95cd-5c5145bcb287
begin
	investigated_model = res.output_model |> device

	vertex_counts = Dict(zip(vertices(G), zeros(length(vertices(G)))))
	vertex_outcomes = Dict(zip(vertices(G), zeros(length(vertices(G)))))
	
	graphlets = MolecularGraphKernels.con_sub_g(3,G)
	
	graph_vec = [induced_subgraph(G,graphlet)[1] for graphlet ∈ graphlets]
	
	featuredgraph_vec = mg_to_fg(graph_vec,res.data.labels.edge_labels,res.data.labels.vertex_labels)
	
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

# ╔═╡ 8499e57e-8e1f-4c76-9650-11c784309569
viz_graph(G)

# ╔═╡ 192db9bf-e3b5-4788-a8bb-325b14ac3165
for v in vertices(G)
	set_prop!(G, v, :alpha, vertex_outcomes[v])
end

# ╔═╡ 8c8f77aa-2394-4abf-93e1-856432b93112
begin
	edge_alpha_G = [(vertex_outcomes[i]+1)/2 for i in vertices(G)]
	viz_graph(G, node_alpha_mask=edge_alpha_G)
end

# ╔═╡ 8205224c-1250-4c9c-b361-a15c2dc17fbf
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
	graph_classes = btx_class_labels[[i ∉[errored_smiles[i][1] for i ∈ 1:length(errored_smiles)] for i in 1:length(btx_class_labels)]]

	btx_mg_bal, btx_class_bal = MLUtils.oversample(btx_graphs,graph_classes)

	labels = RWKLayerFunctions.find_labels(collect(btx_mg_bal))

	btx_featuredgraphs = RWKLayerFunctions.mg_to_fg(collect(btx_mg_bal),labels.edge_labels,labels.vertex_labels)
end

# ╔═╡ a1257336-376b-451e-b0df-55cc34e78d59
@bind bee_tox_i Slider(1:length(btx_graphs))

# ╔═╡ d6e4116a-96e4-41e5-a1fc-814089de711f
begin
	g_btx = btx_mg_bal[bee_tox_i]
	viz_graph(g_btx, layout_style = :molecular)
end

# ╔═╡ dcbd4f11-5bf8-4e2e-87e8-8a451020ba20
md"Is this molecule toxic to honeybees? T/F"

# ╔═╡ 6e808961-0cb5-4843-a893-b9743d1da383
btx_class_bal[bee_tox_i]

# ╔═╡ 8efec125-9d67-4492-966f-85aed4317d62
md"Does the trained model think that it is toxic? T/F"

# ╔═╡ 159eb39c-5556-4e67-b69c-1b1647a9581c
md"Here is the output of the model for this graph:"

# ╔═╡ 5768dafb-c551-4e42-bb06-f17ff696aaac
@load pwd()*"\\results\\opwVkHiDwd.jld2" res_med

# ╔═╡ b6008ccc-4b4e-4489-9433-4f8ce7afd6b0
res_med.output_model.layers[1][1]

# ╔═╡ 3bce473a-d03a-45d5-b8b7-ee0203480c12
m_btx = res_med.output_model|>gpu

# ╔═╡ 3016723e-36b4-4142-a8dd-97bff4cf8196
begin
	preds = m_btx.(btx_featuredgraphs|>gpu)|>cpu
	pred_vector = reduce(hcat,preds)
	pred_bool = [pred_vector[1,i].==maximum(pred_vector[:,i]) for i ∈ 1:size(pred_vector)[2]]
end

# ╔═╡ acfc2998-db0a-4b19-b25e-1ba5e4cac06e
pred_bool[bee_tox_i]

# ╔═╡ cc0d64df-5e78-4332-99eb-f753af91fa88
preds[bee_tox_i]

# ╔═╡ e793dba6-92fe-4b7d-9ae4-3aafbe259f32
begin
	test_data = res_med.data.testing_data
	true_classes = [example[2] for example in test_data]
	pred_classes = [(m_btx(example[1]|>gpu)|>cpu) for example in test_data]
	pred_classes_bool = [i[1]==maximum(i) for i in pred_classes]
end

# ╔═╡ 2a846906-b2bf-4d65-878a-b5f5db263194
begin
	a = plot(res_med.losses, yaxis="Loss")
	ylabel!("Crossentropy Loss")
	xlabel!(a,"Epoch")
	b = twinx(a)
	plot!(b,res_med.epoch_test_accuracy, yaxis = "Accuracy, Recall", linecolor = "light green")
	plot!(b,res_med.epoch_test_recall, linecolor = "yellow")
	ylims!(b,0, 1)
	
	
end

# ╔═╡ fc4ec6e7-35dd-4afb-8541-f061934d7897
@bind hg_med Slider(1:6)

# ╔═╡ 10b58953-1625-4a13-a810-65db70e97a1a
RWKLayerFunctions.hidden_graph_view(res_med.output_model,hg_med)[3]

# ╔═╡ f172d5ed-5ed7-42bc-9cfd-851fef43f478
begin
	hgm1 = hg_to_mg(res_med,hg_med)
	edge_alpha_hgm1 = [get_prop(hgm1,e, :alpha) for e in edges(hgm1)]
	viz_graph(hgm1, edge_alpha_mask=edge_alpha_hgm1,layout_style = :circular)
end

# ╔═╡ 3d5c39cb-9516-49a1-9591-ade3b8d5157b
function gk_contribution_map(res,G)
	investigated_model = res.output_model |> device

	vertex_counts = Dict(zip(vertices(G), zeros(length(vertices(G)))))
	vertex_outcomes = Dict(zip(vertices(G), zeros(length(vertices(G)))))
	
	graphlets = MolecularGraphKernels.con_sub_g(res.output_model.layers[1][1].p,G)
	
	graph_vec = [induced_subgraph(G,graphlet)[1] for graphlet ∈ graphlets]
	
	featuredgraph_vec = mg_to_fg(graph_vec,res.data.labels.edge_labels,res.data.labels.vertex_labels)
	
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
	return vertex_outcomes
end

# ╔═╡ 92dd4d01-58c5-4217-b171-344763af3344
begin
	vertex_contributions = gk_contribution_map(res_med,g_btx)
	btx_alpha_mask = -[(((vertex_contributions[i])/(softmax([1,0])[1]))+1)/2 for i in 1:length(vertex_contributions)].+1
	viz_graph(g_btx, node_alpha_mask=btx_alpha_mask, layout_style =:molecular)
end

# ╔═╡ 59332fe2-25c3-46b1-a8a6-4c0ccfe2bdc2
function model_scores(preds, truths)
	tp = sum(preds.==truths.==1)

	fp = sum(preds.==truths.+1)
	
	tn = sum(preds.==truths.==0)
	
	fn = sum(preds.==truths.-1)
	
	f1 = 2*tp/(2*tp+fp+fn)

	pre = tp/(tp+fp)

	acc = (tp+tn)/(tp+fp+tn+fn)

	rec = tp/(fn+tp)

	return (;pre, rec, acc, f1)
end

# ╔═╡ c1aac3ef-4ef8-4490-9418-50bfb48aa7d4
model_scores(pred_classes_bool, true_classes)

# ╔═╡ Cell order:
# ╠═1ea5cf04-f5e2-488e-882a-18554a3f632f
# ╠═31cf1fe0-d5a2-11ee-0215-07266138b2b6
# ╠═98d0be39-5f65-4a7b-82fd-d99d3869b926
# ╠═59989cc7-7e6f-4330-a926-4cad0a908115
# ╟─0045ba50-90e4-4903-8621-26325bed6c5d
# ╠═184acc05-69c2-4828-b452-f963d5391e13
# ╟─322e22d3-ac2d-4ea1-b043-5bb9ffc0fc84
# ╟─507043b5-5162-4608-b5a1-65e2cb0f2b3c
# ╠═4a8339bc-18b0-46c5-8144-5a2c9a807897
# ╟─1eb98b42-f8cc-4717-b725-e6dfba1d9928
# ╠═4d526687-d354-4176-bbe3-20bcf27f2598
# ╟─9a935611-1efb-48f9-aa18-c8453430d09d
# ╠═48b61aea-6a25-4143-b1be-04b839d43929
# ╟─66f23fa4-3eaf-43e9-930d-99e8e928d74f
# ╠═5990adbd-d3b8-4c03-9f3e-4bfe2b86e1ab
# ╠═5e6915a9-6c0a-4f0c-bd90-d397b9e5ad12
# ╠═bcfbc20c-4587-4efa-8173-f032ac9e609a
# ╠═dc5e40f8-bd39-4769-a524-32144b4fc230
# ╠═49b0c3b8-3b9b-47c0-b640-a0c378b1699b
# ╟─62bd0181-6e87-457e-b8ef-cbd2403cdb58
# ╠═dc37ad80-9e1a-488e-9f9d-6a92e7166d90
# ╠═74327d26-b4ae-4582-a1b6-2797ed1a0360
# ╠═a9655c63-8039-4198-8465-7c80475d93b9
# ╠═b5d13c10-e841-459c-b189-9e6317f8acaf
# ╠═e6f246ab-7d58-4f4e-9e54-134f3d9ea80e
# ╠═2b42e1ae-4b73-4ee4-8318-690ebf0d11fa
# ╠═fd87e5c5-2022-4d06-a9c5-955306eaac2d
# ╠═e50bc67b-4a7c-4084-80e6-d177f15ff967
# ╠═f732df66-75d5-49b7-82d6-54c0f193c369
# ╠═2ae2ec42-f481-4d44-a113-4ca40f1cddc1
# ╠═ee98aa2a-bdb5-4b97-bd6e-d8ba58637077
# ╟─2c6d835b-ba00-4fbc-803b-00d14383e990
# ╠═34ea3bda-2005-4104-85db-61014d634b84
# ╠═24e4f3dd-5399-411a-bdcb-f17d6fff4947
# ╠═58a65a8d-24f9-42e9-8794-b53144d4cae4
# ╠═93e2dedd-e8a0-443e-b691-326a749ebf32
# ╠═b153936e-e321-454a-95cd-5c5145bcb287
# ╠═c40bcbd8-85b6-4f00-b43d-c0db20c003ec
# ╠═8499e57e-8e1f-4c76-9650-11c784309569
# ╠═192db9bf-e3b5-4788-a8bb-325b14ac3165
# ╠═8c8f77aa-2394-4abf-93e1-856432b93112
# ╠═8205224c-1250-4c9c-b361-a15c2dc17fbf
# ╠═a1257336-376b-451e-b0df-55cc34e78d59
# ╠═d6e4116a-96e4-41e5-a1fc-814089de711f
# ╟─dcbd4f11-5bf8-4e2e-87e8-8a451020ba20
# ╟─6e808961-0cb5-4843-a893-b9743d1da383
# ╟─8efec125-9d67-4492-966f-85aed4317d62
# ╟─acfc2998-db0a-4b19-b25e-1ba5e4cac06e
# ╟─159eb39c-5556-4e67-b69c-1b1647a9581c
# ╟─cc0d64df-5e78-4332-99eb-f753af91fa88
# ╠═92dd4d01-58c5-4217-b171-344763af3344
# ╠═5768dafb-c551-4e42-bb06-f17ff696aaac
# ╠═b6008ccc-4b4e-4489-9433-4f8ce7afd6b0
# ╠═3bce473a-d03a-45d5-b8b7-ee0203480c12
# ╠═3016723e-36b4-4142-a8dd-97bff4cf8196
# ╠═e793dba6-92fe-4b7d-9ae4-3aafbe259f32
# ╟─c1aac3ef-4ef8-4490-9418-50bfb48aa7d4
# ╠═2a846906-b2bf-4d65-878a-b5f5db263194
# ╠═fc4ec6e7-35dd-4afb-8541-f061934d7897
# ╠═10b58953-1625-4a13-a810-65db70e97a1a
# ╠═f172d5ed-5ed7-42bc-9cfd-851fef43f478
# ╠═3d5c39cb-9516-49a1-9591-ade3b8d5157b
# ╟─59332fe2-25c3-46b1-a8a6-4c0ccfe2bdc2
