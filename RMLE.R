adaptive_lasso_cd <- function(S, 
                              Y, 
                              lambda, 
                              weights = rep(1, ncol(S)),
                              max_iter = 10000,       
                              tol = 1e-6,            
                              learning_rate = 0.00001,  
                              verbose = TRUE,        
                              initial) { 
  # 输入参数验证 
  stopifnot(nrow(S) == nrow(Y))         
  stopifnot(ncol(Y) == 4)               
  stopifnot(length(weights) == ncol(S)) 
  
  # 初始化参数
  p <- ncol(S)        # dimension of features: p=2+3m
  m <- (p - 2) %/% 3  
  theta <- initial  # use MLE as initialization
  prev_theta <- theta 
  C <- 3  # 前三个类别
  
  map <- vector("list", p)    
  
  # 定义参数结构映射: A_c, the summation indicator 
  map[[1]] <- list(c = 1,   coeff = 1)           # theta1 
  map[[2]] <- list(c = 1:3, coeff = c(2,1,1))    # theta2 
  
  # gamma (3:(2+m))
  for(j in 3:(2+m)) {
    map[[j]] <- list(c = 1:3, coeff = c(2,1,1))
  }
  
  # zeta  ((3+m):(2+2m))
  for(j in (3+m):(2+2*m)) {
    map[[j]] <- list(c = c(1,3), coeff = c(1,1)) #先13 再12，参考4分类的映射
  }
  
  # eta ((3+2m):p)
  for(j in (3+2*m):p) {
    map[[j]] <- list(c = c(1,2), coeff = c(1,1))
  }
  
  A_list <- construct_A_matrices((ncol(S)-2)%/%3)
  
  # 主循环
  for(iter in 1:max_iter) { 
    theta_old <- theta 
    
    for(j in 1:p) { # 遍历每个参数进行更新，coordinate decent 
      
      # 计算当前参数的影响
      grad_theta_j <- function(theta, j) {
        # theta: 原始参数向量 (长度为 p)
        # j: 需要计算梯度的索引
        
        # 提取对应类别的对角矩阵
        A1 <- A_list[[1]]
        A2 <- A_list[[2]]
        A3 <- A_list[[3]]
        
        # 计算各类别的线性预测器 (n x 1)
        eta1 <- S %*% (A1 %*% theta)
        eta2 <- S %*% (A2 %*% theta)
        eta3 <- S %*% (A3 %*% theta)
        
        # 计算softmax分母: D = 1 + exp(eta1) + exp(eta2) + exp(eta3)
        D <- 1 + exp(eta1) + exp(eta2) + exp(eta3)
        
        # 计算各类别概率 (n x 1)
        p1 <- exp(eta1) / D
        p2 <- exp(eta2) / D
        p3 <- exp(eta3) / D
        
        # 对于每个观测 i, 第 j 维的梯度贡献为:
        #   (Y[i,1] - p1[i]) * a1_j * S[i,j] + (Y[i,2] - p2[i]) * a2_j * S[i,j] + (Y[i,3] - p3[i]) * a3_j * S[i,j]
        # 其中 a{c}_j 为 A_c 的第 j 个对角元
        a1_j <- A1[j, j]
        a2_j <- A2[j, j]
        a3_j <- A3[j, j]
        
        grad_j_val <- sum((Y[,1] - p1) * a1_j * S[, j]) +
          sum((Y[,2] - p2) * a2_j * S[, j]) +
          sum((Y[,3] - p3) * a3_j * S[, j])
        
        # 返回负梯度（用于最小化负对数似然）
        return(grad_j_val)
      }
      
      # 软阈值更新 
      z_j <- theta[j] + learning_rate * grad_theta_j(theta,j) 
      threshold <- lambda * weights[j]  * learning_rate 
      theta[j] <- sign(z_j) * pmax(abs(z_j)-threshold, 0) 
    }
    
    # 检查收敛
    delta <- max(abs(theta - theta_old))
    if(verbose && iter %% 50 == 0) {
      cat(sprintf("Iteration %d: max change=%.4f\n", iter, delta))
    }
    if(delta < tol) break
  }
  return(theta)
}
 





# 增强的MLE估计函数（带自适应权重）
regularized_estimator <- function(S, 
                                  Y, 
                                  lambda, 
                                  gamma = 1,
                                  max_iter = 20000, 
                                  tol = 1e-7,
                                  learning_rate = 0.0001, 
                                  verbose = TRUE){  
  A_list <- construct_A_matrices((ncol(S)-2)%/%3)
  # 步骤1：计算初始MLE估计
  mle_theta <- mle_estimator(S, 
                             Y, 
                             construct_A_matrices((ncol(S)-2)%/%3),
                             method = "BFGS",
                             maxit = 80000,
                             theta_init = NULL
  )   
 
  
  # 步骤2：计算自适应权重
  weights <- 1 / (abs(mle_theta)^gamma)
  initial <-  mle_theta
  # 步骤3：运行自适应Lasso
  adalasso_theta <- adaptive_lasso_cd(S, 
                                      Y, 
                                      lambda, 
                                      weights,
                                      max_iter, 
                                      tol, 
                                      learning_rate,  
                                      verbose, 
                                      initial) 
  lasso_theta <- adaptive_lasso_cd(S, 
                                      Y, 
                                      lambda, 
                                      weights=rep(1,ncol(S)),
                                      max_iter, 
                                      tol, 
                                      learning_rate,  
                                      verbose, 
                                      initial)
  # lasso_theta <- data.frame(theta_0, mle_theta, mle_theta2, adalasso_theta) 
  return(list(mle = mle_theta,
              lasso = lasso_theta,
              adalasso = adalasso_theta))
}

 
