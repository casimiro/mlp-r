adaboostr <- function(models)
{

    res = list(models=models, modelsBeta=rep(0,length(models)))
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
    print("Start boosting")
    Ls <- array(0,c(length(adaboostr$models),nrow(x)))
    ps <- array(0,c(length(adaboostr$models),nrow(x)))
    diffs <- array(1,c(length(adaboostr$models),nrow(x)))
    for (model in adaboostr$models)
    {
        ps[curModel,] <- p
        tset <- sample(index, nrow(x), TRUE, p)

        yhat <- sapply(tset, function(j) train(model,x[j,],y[j]))
        diff <- abs(y[tset] - yhat)
        diffs[curModel,tset] <- diff
        L <- diff / max(diff)
        Ls[curModel,] <- L
        lhat <- sum(L*p[tset])
        print("lhat")
        print(lhat)
        modelsBeta[curModel] <- lhat / (1 - lhat)
        p[tset] <- p[tset] * (modelsBeta[curModel])^(1 - L)
        p <- p/sum(p)
        curModel <- curModel + 1
        #if (lhat <= 0.5)
        #    break
    }
    assign('p', p, envir=adaboostr)
    assign('modelsBeta', modelsBeta, envir=adaboostr)
    r <- list(ps=ps,diffs=diffs,Ls=Ls)
    r
}


predict.adaboostr <- function(adaboostr, x)
{
    # para compor as hipoteses dos especialistas,
    # eu devo ordenar suas predicoes de forma crescente
    # e pegar a primeira predicao cuja soma dos log(1/beta) das primeiras
    # predicoes ate ele seja maior que o log(1/beta) medio

    threshold <- mean(log(1/adaboostr$modelsBeta))
    ys <- sapply(adaboostr$models, function(mlp) predict(mlp,x))
    invBetas <- log(1/adaboostr$modelsBeta[sort.list(ys)])
    res <- 0

    for (i in 1:length(ys)) {
        res <- ys[i]
        if (sum(invBetas[1:i]) > threshold)
            break;
    }
    res
}


