using JLD2

@load "C:\\Users\\dcase\\RWKLayer\\res.jld2" res

begin
	plot(res.losses, yaxis="Loss")
	plot!(twinx(),res.epoch_test_accuracy, yaxis = "accuracy", linecolor = "light green")
	plot!(twinx(),res.epoch_test_similarity, linecolor = "yellow")
end