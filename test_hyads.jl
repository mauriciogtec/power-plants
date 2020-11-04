include("lagged.jl")



function main()
    # loop conf
    save_every = 5 # saves model as bson
    print_every = 1
    export_every = 5 # exports kernel params to csv
    niter = 500
    load = false
    suffix = "_unnorm"
    
    # load data
    X, y = load_data()
    
    # build model
    η = 10f-1  # spatio-temporal smoothing
    ν = 1f-1  # power plant effect overall shrinkage
    if !load
        m = LagModel(X, y, η, ν)
    else
        @load "results/model_laplace$suffix.bson" m
        m.ν = ν
        m.η = η
    end

    # step
    for iter in 1:niter
        println("Iteration $iter")
        
        @time loss, tv, ridge = learn!(m);
        println("Starting loss: $loss\t|\ttv: $tv\t|\tridge: $ridge\n") 

        if iter ==1 || iter % print_every == 0
            @printf "γ: (%.4f, %.4f, %.4f) " minimum(m.γ) mean(m.γ) maximum(m.γ)
            @printf "μ: (%.3f, %.3f, %.3f) " minimum(m.μ) mean(m.μ) maximum(m.μ)
            @printf "λ: (%.3f, %.3f, %.3f)\n" minimum(m.λ) mean(m.λ) maximum(m.λ)
        end
        
        if iter % save_every == 0
            @save "results/model_laplace$suffix.bson" m
        end

        if iter % export_every == 0
            export_params_to_csv(m, suffix=suffix)
        end
    end

end

main()
