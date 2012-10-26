create.instances <- function(data, lag=7, testPercentage=0.2)
{
	rows <- dim(data)[1]
	nInst <- rows - lag
	insts <- vector("list", nInst)
	res <- vector("list", nInst)
	for (i in 1:nInst)
	{				
		insts[[i]] <- data[i:(i+lag-1),]
		res[[i]] <- data[(i+lag),]
	}
	nTests <- as.integer(nInst*testPercentage)
	tests <- sample(1:nInst, nTests)
	training <- setdiff(1:nInst, tests)
	ret = list(insts=insts[training], res=res[training], tInsts = insts[tests], tRes=res[tests])
	ret
}

calculate.diff <- function()

neuralnet <- function(data, testData, nh=5, lr=0.1, maxSeasons=500, targetErr=0.01)
{
	ni <- length(data$insts[[1]])
	no <- length(data$res[[1]])
	seasons <- 0
	
	A <- replicate(nh, rnorm(ni+1))
	A_diff <- array(1, c(ni+1,nh))
	B <- replicate(no, rnorm(nh+1))
	B_diff <- array(1, c(nh+1, no))
	converted <- FALSE
	meanErr <- 0
	while(!converted) {
		
		# training
		for (i in 1:length(data$insts))
		{
			inWithBias <- c(data$insts[[i]],1)
			outHid <- inWithBias %*% A
			zOutHid <- sapply(outHid, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			netOut <- c(zOutHid,1) %*% B
			zNetOut <- (exp(netOut) - exp(-netOut))/(exp(netOut) + exp(-netOut))
			
			err = (zNetOut - data$res[[i]])
			outDelta <- lr*(1 - zNetOut^2)* err * c(zOutHid,1) 
			
			for (j in 1:(ni+1))
				for (k in 1:nh)
					A_diff[j,k] <- lr*(1 - netOut^2)* B[k] * (1 - zOutHid[k]^2) * err * inWithBias[j]
			
			A <- A - A_diff
			B <- B - outDelta
		}
		
		meanErr <- 0
		#testing
		for (i in 1:length(data$tInsts)) {
			outHid <- sapply(c(data$tInsts[[i]],1) %*% A, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			netOut <- sapply(c(outHid,1) %*% B, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			err <- (netOut - data$tRes[[i]])^2/2
			meanErr <- meanErr + err
		}
		meanErr <- meanErr / length(data$tInsts)
		print (meanErr)
		if (meanErr < targetErr || seasons == maxSeasons )
			converted <- TRUE
	}
	list(seasons=seasons, erro=meanErr)
	
}
