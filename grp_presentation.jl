### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 31cf1fe0-d5a2-11ee-0215-07266138b2b6
begin
	import Pkg
	Pkg.activate(".");
	using MetaGraphs, CSV, DataFrames,MolecularGraph, JLD2, MolecularGraphKernels, GeometricFlux, Flux, CUDA, RWKLayerFunctions, Graphs, Plots, PlutoUI, LinearAlgebra
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

"

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

# ╔═╡ 74327d26-b4ae-4582-a1b6-2797ed1a0360
begin # isomorphism learning test with one size 4 hidden graph, loss is 1-kernel score
	A_2 = Float32.(rand(4,4, 4))
	model = RWKLayerFunctions.KGNN(A_2, Float32.(rand(4,3)), 3, 4, relu) |> device
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

# ╔═╡ dc37ad80-9e1a-488e-9f9d-6a92e7166d90
RWKLayerFunctions.KGNN(A_2, Float32.(rand(4,3)), 3, 4, relu)

# ╔═╡ b5d13c10-e841-459c-b189-9e6317f8acaf
begin
	plot(-(losses).+1, linewidth=3)
	ylabel!("Kernel Score (Cosine Norm)")
	xlabel!("Epoch")
end

# ╔═╡ 2b42e1ae-4b73-4ee4-8318-690ebf0d11fa
begin # same logic as inside the kernel layer, to replicate the hidden graph from the "point of view" of the kernel function
	mcpu = (model |> cpu)
	
	nv = size(mcpu.h_adj)[1]
		
	n_edge_types = size(mcpu.h_adj)[3]

	A_norm_mx = RWKLayerFunctions.upper_triang(nv)
		
	A_2_herm = stack([relu.((mcpu.h_adj[:,:,i].*A_norm_mx+(mcpu.h_adj[:,:,i].*A_norm_mx)')) for i ∈ 1:4])

	id = Matrix{Float32}(I, nv, nv) |> (isa(mcpu.h_adj, CuArray) ? gpu : cpu)  
	
	A_2_adj = stack([sum([A_2_herm[:,:,i] for i ∈ 1:n_edge_types]).+id for i ∈ 1:n_edge_types])
	
	A_2_norm = A_2_herm./A_2_adj

	col_to_label = Dict(zip(1:3, 7:9))

	slice_to_label = Dict(zip(1:3, 1:3))
	
	X_2_norm = relu.(mcpu.h_nf)
	
	# get adjacency matrix by edge thresholding
	A = sum([A_2_norm[:,:,i].>.5 for i in axes(A_2_norm[:,:,1:end-1], 3)])

	v_props = []
	for r in 1:size(A_2_norm)[1]

		push!(v_props,col_to_label[argmax(X_2_norm[r,:])])

	end

	# generate graph topology
	mg = MetaGraph(SimpleGraph(A))

	for v in vertices(mg)
		set_prop!(mg, v, :label, v_props[v])
	end

	# set edge weights
	wts = reshape(
		[slice_to_label[x[3]] for x in argmax(A_2_norm[:,:,1:end-1]; dims=3)], 
		axes(A_2_norm[:, :, 1])
	)

	alphas = reshape(
		[x for x in maximum(A_2_norm[:,:,1:end-1]; dims=3)], 
		axes(A_2_norm[:, :, 1])
	)
	for e in edges(mg)
		set_prop!(mg, e, :label, wts[src(e), dst(e)])
		set_prop!(mg, e, :alpha, alphas[src(e), dst(e)])
	end

	edge_alpha_mg = [get_prop(mg,e, :alpha) for e in edges(mg)]
	viz_graph(mg, edge_alpha_mask=edge_alpha_mg, )
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
@load "C:\\Users\\dcase\\RWKLayer\\results\\8zSbqXc5wr.jld2" res

# ╔═╡ ee98aa2a-bdb5-4b97-bd6e-d8ba58637077
begin
	plot(res.losses, linewidth=3)
	ylabel!("Crossentropy Loss")
	xlabel!("Epoch")
	
end

# ╔═╡ 2c6d835b-ba00-4fbc-803b-00d14383e990
md"`train_kgnn(synthetic_graph_data[5:40],graph_labels[5:40], lr = .001, n_epoch = 300, n_hg = 4, batch_sz = 1, p=3, size_hg = 4, train_portion = .95)`"

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
	
	graphlets = MolecularGraphKernels.con_sub_g(4,G)
	
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

# ╔═╡ 5768dafb-c551-4e42-bb06-f17ff696aaac
@load "C:\\Users\\dcase\\RWKLayer\\results\\fK7C5oVSHI.jld2" res_med

# ╔═╡ 8205224c-1250-4c9c-b361-a15c2dc17fbf
begin # load synthetic test set #1
	synthetic_data_med = load("C:\\Users\\dcase\\RWKLayer\\molecules_med.jld2")
	synthetic_graph_data_med = synthetic_data_med["molecules"]
	synthetic_classes_med = synthetic_data_med["class_enc"]
end

# ╔═╡ 1cd197b7-a105-491b-92b5-0704996225b5
viz_graph(synthetic_graph_data_med[7])

# ╔═╡ 2a846906-b2bf-4d65-878a-b5f5db263194
begin
	plot(res_med.losses, linewidth=3)
	ylabel!("Crossentropy Loss")
	xlabel!("Epoch")
end

# ╔═╡ f172d5ed-5ed7-42bc-9cfd-851fef43f478
begin
	hgm1 = hg_to_mg(res_med,1)
	edge_alpha_hgm1 = [get_prop(hgm1,e, :alpha) for e in edges(hgm1)]
	viz_graph(hgm1, edge_alpha_mask=edge_alpha_hgm1)
end

# ╔═╡ 0d2c18b2-03e2-4a5a-ac7e-2a33814c93f4
begin
	hgm2 = hg_to_mg(res_med,2)
	edge_alpha_hgm2 = [get_prop(hgm2,e, :alpha) for e in edges(hgm2)]
	viz_graph(hgm2, edge_alpha_mask=edge_alpha_hgm2)
end

# ╔═╡ fbad8f6f-f255-489b-867e-e7e09412ffb0
begin
	hgm3 = hg_to_mg(res_med,2)
	edge_alpha_hgm3 = [get_prop(hgm3,e, :alpha) for e in edges(hgm3)]
	viz_graph(hgm3, edge_alpha_mask=edge_alpha_hgm3)
end

# ╔═╡ a62a8585-533b-470c-96a1-92e61eb0a55d
begin
	hgm4 = hg_to_mg(res_med,2)
	edge_alpha_hgm4 = [get_prop(hgm4,e, :alpha) for e in edges(hgm4)]
	viz_graph(hgm4, edge_alpha_mask=edge_alpha_hgm4)
end

# ╔═╡ 567d9a59-b829-4ebe-a8f3-868eebc5f82f
begin
	hgm5 = hg_to_mg(res_med,2)
	edge_alpha_hgm5 = [get_prop(hgm5,e, :alpha) for e in edges(hgm5)]
	viz_graph(hgm5, edge_alpha_mask=edge_alpha_hgm5)
end

# ╔═╡ 66a85a90-2616-4b03-809d-d93119b5f046
begin
	hgm6 = hg_to_mg(res_med,2)
	edge_alpha_hgm6 = [get_prop(hgm6,e, :alpha) for e in edges(hgm6)]
	viz_graph(hgm6, edge_alpha_mask=edge_alpha_hgm6)
end

# ╔═╡ Cell order:
# ╠═1ea5cf04-f5e2-488e-882a-18554a3f632f
# ╠═31cf1fe0-d5a2-11ee-0215-07266138b2b6
# ╠═98d0be39-5f65-4a7b-82fd-d99d3869b926
# ╠═59989cc7-7e6f-4330-a926-4cad0a908115
# ╠═0045ba50-90e4-4903-8621-26325bed6c5d
# ╠═184acc05-69c2-4828-b452-f963d5391e13
# ╟─322e22d3-ac2d-4ea1-b043-5bb9ffc0fc84
# ╟─507043b5-5162-4608-b5a1-65e2cb0f2b3c
# ╟─4a8339bc-18b0-46c5-8144-5a2c9a807897
# ╟─1eb98b42-f8cc-4717-b725-e6dfba1d9928
# ╟─4d526687-d354-4176-bbe3-20bcf27f2598
# ╟─9a935611-1efb-48f9-aa18-c8453430d09d
# ╟─48b61aea-6a25-4143-b1be-04b839d43929
# ╠═66f23fa4-3eaf-43e9-930d-99e8e928d74f
# ╠═5990adbd-d3b8-4c03-9f3e-4bfe2b86e1ab
# ╠═5e6915a9-6c0a-4f0c-bd90-d397b9e5ad12
# ╠═bcfbc20c-4587-4efa-8173-f032ac9e609a
# ╠═dc5e40f8-bd39-4769-a524-32144b4fc230
# ╠═49b0c3b8-3b9b-47c0-b640-a0c378b1699b
# ╠═dc37ad80-9e1a-488e-9f9d-6a92e7166d90
# ╠═74327d26-b4ae-4582-a1b6-2797ed1a0360
# ╠═b5d13c10-e841-459c-b189-9e6317f8acaf
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
# ╠═c40bcbd8-85b6-4f00-b43d-c0db20c003ec
# ╠═b153936e-e321-454a-95cd-5c5145bcb287
# ╠═8499e57e-8e1f-4c76-9650-11c784309569
# ╠═192db9bf-e3b5-4788-a8bb-325b14ac3165
# ╠═8c8f77aa-2394-4abf-93e1-856432b93112
# ╠═5768dafb-c551-4e42-bb06-f17ff696aaac
# ╠═8205224c-1250-4c9c-b361-a15c2dc17fbf
# ╠═1cd197b7-a105-491b-92b5-0704996225b5
# ╠═2a846906-b2bf-4d65-878a-b5f5db263194
# ╠═f172d5ed-5ed7-42bc-9cfd-851fef43f478
# ╠═0d2c18b2-03e2-4a5a-ac7e-2a33814c93f4
# ╠═fbad8f6f-f255-489b-867e-e7e09412ffb0
# ╠═a62a8585-533b-470c-96a1-92e61eb0a55d
# ╠═567d9a59-b829-4ebe-a8f3-868eebc5f82f
# ╠═66a85a90-2616-4b03-809d-d93119b5f046
