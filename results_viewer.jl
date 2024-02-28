begin
	import Pkg
	Pkg.activate()
	using MetaGraphs, CSV, DataFrames,MolecularGraph, JLD2, MolecularGraphKernels, GeometricFlux, Flux, CUDA, Plots
end

include("C:\\Users\\dcase\\RWKLayer\\src\\RWKLayerFunctions.jl")

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

# begin # load synthetic test set #1
# 	synthetic_data = load("C:\\Users\\dcase\\RWKLayer\\molecules_med.jld2")
# 	synthetic_graph_data = synthetic_data["molecules"]
# 	synthetic_classes = synthetic_data["class_enc"]
# end

#graph_labels = [get_prop(g, :label) for g in synthetic_graph_data]


res_med = train_kgnn(btx_featuredgraphs,graph_classes, lr = .01, n_epoch = 1000, n_hg = 6, batch_sz = 2, p=3, size_hg = 5);


#RWKLayerFunctions.hidden_graph_view(res.output_model, 1)[1]

#@load "C:\\Users\\dcase\\RWKLayer\\res.jld2" res

filename = "C:\\Users\\dcase\\RWKLayer\\results\\" * tempname()[end-9:end]*".jld2"

#model = res.output_model |> gpu
jldsave(filename, compress=false; res_med)
plot(res_med.losses)
#RWKLayerFunctions.hidden_graph_view(res.output_model, 1)

#column_labels = Dict(zip(1:length(res.data.labels.vertex_labels), res.data.labels.vertex_labels))
#slice_labels = Dict(zip(1:length(res.data.labels.edge_labels), res.data.labels.edge_labels))

#begin
	#plot(res.losses, yaxis="Loss")
	#plot!(twinx(),res.epoch_test_accuracy, yaxis = "accuracy", linecolor = "light green")
	#plot!(twinx(),res.epoch_test_recall, linecolor = "yellow")
#end

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
