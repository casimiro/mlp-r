require(foreach)
require(doMC)

create.instances <- function(data, lag=7, testPercentage=0.2)
{
	rows <- dim(data)[1]
	nInst <- rows - lag
	insts <- array(NA, c(nInst,lag))#vector("list", nInst)
	res <- rep(NA, nInst)
	for (i in 1:nInst)
	{
		insts[i,] <- data[i:(i+lag-1),]
		res[i] <- data[(i+lag),]
	}
	nTests <- as.integer(nInst*testPercentage)
	tests <- sample(1:nInst, nTests)
	training <- setdiff(1:nInst, tests)
	ret = list(insts=insts[training,], res=res[training], tInsts = insts[tests,], tRes=res[tests])
	ret
}

cross.validation <- function(m, x, y, k=10)
{
	index <- 1:nrow(x)
	index <- sample(index,nrow(x))
	foldSize <- nrow(x) / k
	
	registerDoMC(cores=8)
	errors <- foreach(i = 1:k, .combine='c') %dopar% {
		model <- clone(m)
		testSet <- index[(1+(k-1)*foldSize):(k*foldSize)]
		trainSet <- setdiff(index,testSet)
		if(class(model) == "adaboostr")
			train(model,x[trainSet,],y[trainSet])
		else # mlp
			sapply(trainSet, function(j) train(model,x[j,], y[j]))
		yhat <- sapply(testSet, function(j) predict(model, x[j,]))
		diff <- y[testSet] - yhat
		
		mean(diff^2)
		
	}
	
	
	errors
}

