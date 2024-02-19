### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ b2680020-cf69-11ee-19c3-45d6c00b88f9
begin
	import Pkg
	Pkg.activate()
	using MetaGraphs, Flux, Functors, Zygote, LinearAlgebra, Plots, GeometricFlux, PlutoUI, JLD2, Graphs, Random, CUDA, Statistics, CSV, DataFrames, MolecularGraph, MolecularGraphKernels, BenchmarkTools, Distributed
end

# ╔═╡ e768eac5-96de-408c-ae98-435b647b9bcc
begin
	using Main.var"workspace#4".RWKLayerFunctions
	#@load "C:\\Users\\dcase\\RWKLayer\\res.jld2" res
end

# ╔═╡ b13ee9ff-da26-47ab-9f3c-1edfacbf198e
include("C:\\Users\\dcase\\RWKLayer\\RWKLayerFunctions.jl")

# ╔═╡ aa36ddd5-1664-41c9-a0bf-ad87b977383b
begin
	plot(res.losses, yaxis="Loss")
	plot!(twinx(),res.epoch_test_accuracy, yaxis = "accuracy", linecolor = "light green")
	#plot!(twinx(),res.epoch_test_recall, linecolor = "yellow")
end

# ╔═╡ Cell order:
# ╠═b2680020-cf69-11ee-19c3-45d6c00b88f9
# ╠═b13ee9ff-da26-47ab-9f3c-1edfacbf198e
# ╠═e768eac5-96de-408c-ae98-435b647b9bcc
# ╠═aa36ddd5-1664-41c9-a0bf-ad87b977383b
