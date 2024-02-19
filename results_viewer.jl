begin
	import Pkg
	Pkg.activate()
	using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, PlutoUI, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels, BenchmarkTools, Distributed
	TableOfContents(title="Random walk layer w/ edge labels")
end

include("C:\\Users\\dcase\\RWKLayer\\RWKLayerFunctions.jl")

using .RWKLayerFunctions

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


res = train_kgnn(btx_graphs,graph_classes, lr = 1, n_epoch = 400, n_hg = 12, batch_sz = 24)

#@load "C:\\Users\\dcase\\RWKLayer\\res.jld2" res



#model = res.output_model |> gpu

#RWKLayerFunctions.hidden_graph_view(res.output_model, 1)

column_labels = Dict(zip(1:length(res.data.labels.vertex_labels), res.data.labels.vertex_labels))
slice_labels = Dict(zip(1:length(res.data.labels.edge_labels), res.data.labels.edge_labels))

begin
	plot(res.losses, yaxis="Loss")
	plot!(twinx(),res.epoch_test_accuracy, yaxis = "accuracy", linecolor = "light green")
	plot!(twinx(),res.epoch_test_recall, linecolor = "yellow")
end

#g = RWKLayerFunctions.hidden_graph2(res.output_model, 1, column_labels, slice_labels, 0.5, .3)

#testing_graphs = getindex.(res.data.testing_data,1)
#testing_classes = getindex.(res.data.testing_data,2)

#preds = model.(testing_graphs|>gpu)|>cpu
#pred_vector = reduce(hcat,preds)
#pred_bool = [pred_vector[1,i].==maximum(pred_vector[:,i]) for i ∈ 1:size(pred_vector)[2]]

#tp = sum(pred_bool.==testing_classes.==1)

#fp = sum(pred_bool.==testing_classes.+1)

#tn = sum(pred_bool.==testing_classes.==0)

#fn = sum(pred_bool.==testing_classes.-1)

#f1 = 2*tp/(2*tp+fp+fn)