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


