adaboostr <- function(models)
{
	
    res = list(models=models, modelsBeta=rep(0,length(models)))
	res <- list2env(res)
    class(res) <- 'adaboostr'
    res
}

reset.adaboostr <- function(adaboostr)
{
	for (mlp in adaboostr$models)
		reset(mlp)
	
	modelsBeta=rep(0,length(adaboostr$models))
	assign('modelsBeta', modelsBeta, envir=adaboostr)
}

train.adaboostr <- function(adaboostr, x, y)
{
	reset(adaboostr)
	# probabilitie of each training instance
	p <- rep(1/nrow(x), nrow(x))
	index <- 1:nrow(x)
    modelsBeta <- adaboostr$modelsBeta
	p <- adaboostr$p
    curModel <- 1
    for (mlp in adaboostr$models)
    {
        tset <- sample(index, nrow(x), TRUE, p)
        yhat <- sapply(tset, function(j) train(mlp,x[j,],y[j]))
        diff <- abs(yhat - y[tset])
        L <- diff / max(diff)
        lhat <- sum(L*p[tset])
        modelsBeta[curModel] <- lhat / (1 - lhat)
        p[tset] <- p[tset] * modelsBeta[curModel]^(1 - L)
		curModel <- curModel + 1
        #if (lhat <= 0.5)
        #    break
    }
	assign('p', p, envir=adaboostr)
	assign('modelsBeta', modelsBeta, envir=adaboostr)
}

predict.adaboostr <- function(adaboostr, x)
{
    # para compor as hipoteses dos especialistas,
    # eu devo ordenar suas predicoes de forma crescente
    # e pegar a primeira predicao cuja soma dos log(1/beta) das primeiras
    # predicoes ate ele seja maior que o log(1/beta) medio
  
    threshold <- mean(log(1/adaboostr$modelsBeta))
    ys <- sapply(adaboostr$models, function(mlp) predict(mlp,x))
    invBetas <- adaboostr$modelsBeta[sort(ys)]
    res <- 0
    
    for (i in 1:length(ys)) {
        res <- ys[i]
        if (sum(ys[1:i]) > threshold)
            break;
    }
    res
}

