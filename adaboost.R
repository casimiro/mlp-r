adaboostr <- function(models, x, y, tx, ty)
{
	p <- rep(1/nrows(inputs), nrows(inputs))
    res = list(models=models, x=x, y=y, tx=tx, ty=ty, modelsBeta=rep(0,length(models)), p=p)
    class(res) <- 'adaboostr'
    res
}

train.adaboostr <- function(adaboostr)
{
	# probabilitie of each training instance
	index <- 1:nrows(adaboostr$x)
    modelsBeta <- rep(0,length(adaboostr$models))
    curModel <- 1
    for (mlp in adaboostr$models)
    {
        tset <- sample(index, nrows(adaboostr$x), TRUE, adaboostr$p)
        yhat <- sapply(tset, function(x) train(mlp,adaboostr$x[tset,],adaboostr$y[tset]))
        diff <- abs(yhat - outputs[tset])
        L <- diff / max(diff)
        lhat <- sum(L*adaboostr$p[tset])
        adaboostr$modelsBeta[curModel] <- lhat / (1 - lhat)
        adaboostr$p[tset] <- adaboostr$p[tset] * adaboostr$modelsBeta[curModel]^(1 - L)

        if (lhat <= 0.5)
            break
    }
}

predict.adaboostr <- function(adaboostr)
{
    tyhat <- rep(0, length(adaboostr$ty))
    for (i in 1:length(adaboostr$tx))
    {

    }
}
