### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ b2680020-cf69-11ee-19c3-45d6c00b88f9
begin
	import Pkg
	Pkg.activate("."; io=devnull)
	using MetaGraphs, CSV, DataFrames,MolecularGraph, JLD2, MolecularGraphKernels, RWKLayerFunctions, Flux
end

# ╔═╡ c766a647-1025-45a0-ae4b-325136057bc0
using Plots

# ╔═╡ 190246a5-2fb7-4d84-a309-998386e9a094
Main.RWKLayerFunctions = RWKLayerFunctions

# ╔═╡ e768eac5-96de-408c-ae98-435b647b9bcc
begin
	@load "C:\\Users\\dcase\\RWKLayer\\results\\xs12r2v0Z2.jld2" res
end

# ╔═╡ 31844498-5c2c-4555-b283-f1706985b828
res.losses

# ╔═╡ aa36ddd5-1664-41c9-a0bf-ad87b977383b
begin
	plot(res.losses, yaxis="Loss")
	plot!(twinx(),res.epoch_test_accuracy, yaxis = "accuracy", linecolor = "light green")
	plot!(twinx(),res.epoch_test_recall, linecolor = "yellow")
end

# ╔═╡ 7810d7db-46a8-47e1-a933-1e38f45cc32a
res.output_model[1][1].a(0)

# ╔═╡ d95c1fc5-5465-4aa1-aa99-01cc2a48ad54
begin
	column_labels = Dict(zip(1:length(res.data.labels.vertex_labels), res.data.labels.vertex_labels))
	slice_labels = Dict(zip(1:length(res.data.labels.edge_labels), res.data.labels.edge_labels))
end


# ╔═╡ e6df93ba-b7f5-4f56-bb33-d26c9d116c5e
viz_graph(RWKLayerFunctions.hidden_graph2(res.output_model, 1, column_labels, slice_labels, 0.5,.3)[1])

# ╔═╡ a7d7cb75-ca2a-4a81-af11-084a6bb4bce6
RWKLayerFunctions.hidden_graph_view(res.output_model, 1)

# ╔═╡ 52f2491c-9c0f-4877-9b1f-74770b1a357b
res.data.training_data

# ╔═╡ dc285ce3-1b15-4fcd-bd29-4583f71f3016
RWKLayerFunctions.KGNN(
    Float32.(rand(4, 4, 4)), 
    Float32.(rand(4,4)), 
    Int(4), 
    Int(4), 
    relu
).σ(-1)


# ╔═╡ Cell order:
# ╠═b2680020-cf69-11ee-19c3-45d6c00b88f9
# ╠═190246a5-2fb7-4d84-a309-998386e9a094
# ╠═e768eac5-96de-408c-ae98-435b647b9bcc
# ╠═31844498-5c2c-4555-b283-f1706985b828
# ╠═c766a647-1025-45a0-ae4b-325136057bc0
# ╠═aa36ddd5-1664-41c9-a0bf-ad87b977383b
# ╠═7810d7db-46a8-47e1-a933-1e38f45cc32a
# ╠═d95c1fc5-5465-4aa1-aa99-01cc2a48ad54
# ╠═e6df93ba-b7f5-4f56-bb33-d26c9d116c5e
# ╠═a7d7cb75-ca2a-4a81-af11-084a6bb4bce6
# ╠═52f2491c-9c0f-4877-9b1f-74770b1a357b
# ╠═dc285ce3-1b15-4fcd-bd29-4583f71f3016
