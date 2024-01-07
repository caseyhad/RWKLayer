using JLD2, Plots

@load "C:\\Users\\dcase\\RWKLayer\\btx_losses_cpu.jld2" btx_losses_cpu

plot(btx_losses_cpu)