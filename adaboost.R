adaboostr <- function(models, x, y, tx, ty)
{
	p <- rep(1/nrows(inputs), nrows(inputs))
    res = list(models=models, x=x, y=y, tx=tx, ty=ty, modelsBeta=rep(0,length(models)), p=p)
	res <- list2env(res)
    class(res) <- 'adaboostr'
    res
}

train.adaboostr <- function(adaboostr)
{
	# probabilitie of each training instance
	index <- 1:nrows(adaboostr$x)
    modelsBeta <- adaboostr$modelsBeta
	p <- adaboostr$p
    curModel <- 1
    for (mlp in adaboostr$models)
    {
        tset <- sample(index, nrows(adaboostr$x), TRUE, adaboostr$p)
        yhat <- sapply(tset, function(x) train(mlp,adaboostr$x[tset,],adaboostr$y[tset]))
        diff <- abs(yhat - outputs[tset])
        L <- diff / max(diff)
        lhat <- sum(L*adaboostr$p[tset])
        modelsBeta[curModel] <- lhat / (1 - lhat)
        p[tset] <- adaboostr$p[tset] * adaboostr$modelsBeta[curModel]^(1 - L)

        if (lhat <= 0.5)
            break
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
