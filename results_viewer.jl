begin
	import Pkg
	Pkg.activate()
	using MetaGraphs, CSV, DataFrames,MolecularGraph, JLD2, MolecularGraphKernels, GeometricFlux, Flux, CUDA, Plots, MLUtils
end

include("C:\\Users\\dcase\\RWKLayer\\src\\RWKLayerFunctions.jl")

using .RWKLayerFunctions

begin #Training Data Preparation
	data = CSV.read(Base.download("https://github.com/SimonEnsemble/graph-kernel-SVM-for-toxicity-of-pesticides-to-bees/raw/main/BeeToxAI%20Data/File%20S1%20Acute%20contact%20toxicity%20dataset%20for%20classification.csv"),DataFrame)

	btx_class_labels = [data[i,:Outcome] == "Toxic" for i ∈ 1:length(data[:,:Outcome])]


	errored_smiles = []
	btx_graphs = Vector{MetaGraphs.MetaGraph{Int64, Float64}}()
	#disallowed_features = ['.','+']

	for z ∈ 1:length(data[!,:SMILES])
		smiles_string = data[z,:SMILES]
		try
			mol = smilestomol(smiles_string)
			push!(btx_graphs,MetaGraph(mol))

		catch e
			push!(errored_smiles, [z,smiles_string])
		end
	end

	labels = RWKLayerFunctions.find_labels(btx_graphs)

	btx_featuredgraphs = RWKLayerFunctions.mg_to_fg(btx_graphs,labels.edge_labels,labels.vertex_labels)

	graph_classes = btx_class_labels[[i ∉[errored_smiles[i][1] for i ∈ 1:length(errored_smiles)] for i in 1:length(btx_class_labels)]]

	btx_mg_bal, btx_class_bal = collect.(MLUtils.oversample(btx_graphs,graph_classes))

end


# begin # load synthetic test set #1
#  	synthetic_data = load("C:\\Users\\dcase\\RWKLayer\\molecules.jld2")
#  	synthetic_graph_data = synthetic_data["molecules"]
#  	synthetic_classes = synthetic_data["class_enc"]
# end

# graph_labels = [get_prop(g, :label) for g in synthetic_graph_data]

# replicates = []


#res_med = train_kgnn(btx_graphs,graph_classes, lr = .01, n_epoch = 500, n_hg = 6, batch_sz = 1, p=2, size_hg = 6) - might be inconsistant
#train_kgnn(btx_graphs,graph_classes, lr = .001, n_epoch = 1, n_hg = 6, batch_sz = 1, p=3, size_hg = 6, premade_model = res1.output_model) - .81 f1
#train_kgnn(btx_graphs,graph_classes, lr = .1, n_epoch = 200, n_hg = 6, batch_sz = 4, p=3, size_hg = 6) - fairly consistant at ~70 acc
#train_kgnn(btx_graphs,graph_classes, lr = .1, n_epoch = 120, n_hg = 8, batch_sz = 2, p=3, size_hg = 6) - good for feature map for two-stage learning
#good parameters ^^

res_med = train_kgnn(btx_mg_bal,btx_class_bal, lr = .01, n_epoch = 80, n_hg = 6, batch_sz = 1, p=3, size_hg = 6, two_stage = true)
#res_med = train_kgnn(btx_mg_bal,btx_class_bal, lr = .01, n_epoch = 130, batch_sz = 1, premade_model = res1.output_model, freeze_fm = true)

# for test in 1:3
# 	res_med = train_kgnn(synthetic_graph_data,graph_labels, lr = .001, n_epoch = 20, n_hg = 6, batch_sz = 2, p=3, size_hg = 5);
# 	filename = "C:\\Users\\dcase\\RWKLayer\\results\\" * tempname()[end-9:end]*".jld2"
# 	jldsave(filename, compress=false; res_med)
# 	push!(replicates, res_med.losses)
# end

# minimum.(replicates)

#RWKLayerFunctions.hidden_graph_view(res.output_model, 1)[1]

#@load "C:\\Users\\dcase\\RWKLayer\\res.jld2" res

filename = "C:\\Users\\dcase\\RWKLayer\\results\\" * tempname()[end-9:end]*".jld2"

#model = res.output_model |> gpu
jldsave(filename, compress=false; res_med)

#plot(res_med.losses)
#RWKLayerFunctions.hidden_graph_view(res.output_model, 1)

#column_labels = Dict(zip(1:length(res.data.labels.vertex_labels), res.data.labels.vertex_labels))
#slice_labels = Dict(zip(1:length(res.data.labels.edge_labels), res.data.labels.edge_labels))

# begin
# 	a = plot(res_med.losses, yaxis="Loss")
# 	b = twinx(a)
# 	plot!(b,res_med.epoch_test_accuracy, yaxis = "accuracy", linecolor = "light green")
# 	plot!(b,res_med.epoch_test_recall, linecolor = "yellow")
# 	ylims!(b,0, 1)
# end

#g = RWKLayerFunctions.hidden_graph2(res.output_model, 1, column_labels, slice_labels, 0.5, .3)

#testing_graphs = getindex.(res.data.testing_data,1)
#testing_classes = getindex.(res.data.testing_data,2)

#preds = model.(testing_graphs|>gpu)|>cpu
#pred_vector = reduce(hcat,preds)
#pred_bool = [pred_vector[1,i].==maximum(pred_vector[:,i]) for i ∈ 1:size(pred_vector)[2]]

# tp = sum(pred_bool.==testing_classes.==1)

# fp = sum(pred_bool.==testing_classes.+1)

# tn = sum(pred_bool.==testing_classes.==0)

# fn = sum(pred_bool.==testing_classes.-1)

# f1 = 2*tp/(2*tp+fp+fn)
