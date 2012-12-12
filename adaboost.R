adaboostr <- function(models)
{
    res = list(models=models, modelsBeta=rep(0,length(models)), modelsLength=0)
	res <- list2env(res)
    class(res) <- 'adaboostr'
    res
}

clone.adaboostr <- function(adaboostr)
{
  models <- sapply(adaboostr$models, function(m) clone(m))
  c <- adaboostr(models)
  c
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
    # probability of each training instance
    p <- rep(1/nrow(x), nrow(x))
    index <- 1:nrow(x)
    modelsBeta <- adaboostr$modelsBeta
    curModel <- 1
    diffs <- array(1,c(length(adaboostr$models),nrow(x)))
    for (model in adaboostr$models)
    {
        tset <- sample(index, nrow(x), TRUE, p)
        yhat <- sapply(tset, function(j) train(model,x[j,],y[j]))
        diff <- abs(y[tset] - yhat)
        L <- diff / max(diff)
        lhat <- sum(L*p[tset])
        modelsBeta[curModel] <- lhat / (1 - lhat)
        p[tset] <- p[tset] * (modelsBeta[curModel])^(1 - L)
        p <- p/sum(p)
        
        if (lhat <= 0.5)
            break
        curModel <- curModel + 1
    }
	
	assign('modelsLength', curModel, envir=adaboostr)
    assign('p', p, envir=adaboostr)
    assign('modelsBeta', modelsBeta, envir=adaboostr)
}


predict.adaboostr <- function(adaboostr, x)
{
    # para compor as hipoteses dos especialistas,
    # eu devo ordenar suas predicoes de forma crescente
    # e pegar a primeira predicao cuja soma dos log(1/beta) das primeiras
    # predicoes ate ele seja maior que o log(1/beta) medio

    threshold <- mean(log(1/adaboostr$modelsBeta[1:adaboostr$modelsLength]))
    ys <- sapply(1:adaboostr$modelsLength, function(i) predict(adaboostr$models[[i]],x))
    invBetas <- log(1/adaboostr$modelsBeta[sort.list(ys)])
    res <- 0

    for (i in 1:length(ys)) {
        res <- ys[i]
        if (sum(invBetas[1:i]) > threshold)
            break;
    }
    res
}


