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
			inputRow <- c(data$insts[[i]],1)
			output <- inputRow %*% A
			hidden <- sapply(output, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			output <- c(hidden,1) %*% B
			output <- (exp(output) - exp(-output))/(exp(output) + exp(-output))
			
			outDelta <- lr*(1 - output^2)* (output - data$res[[i]]) * c(hidden,1) 
			
			for (j in 1:(ni+1))
				for (k in 1:nh)
					A_diff[j,k] <- lr*(1 - output^2)* B[k] * (1 - hidden[k]^2) * (output - data$res[[i]]) * inputRow[j]
			
			
			A <- A + A_diff
			B <- B + outDelta
			
		}

		meanErr <- 0
		#testing
		for (i in 1:length(data$tInsts)) {
			hidden <- sapply(c(data$tInsts[[i]],1) %*% A, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			output <- sapply(c(hidden,1) %*% B, function(x) {(exp(x) - exp(-x))/(exp(x) + exp(-x))})
			err <- (output - data$tRes[[i]])^2/2
			meanErr <- meanErr + err
		}
		meanErr <- meanErr / length(data$tInsts)
		print (meanErr)
		if (meanErr < targetErr || seasons == maxSeasons )
			converted <- TRUE
	}
	list(seasons=seasons, erro=meanErr)
	
}