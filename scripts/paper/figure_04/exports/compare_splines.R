# Bounded exploratory full-window comparison. Labels and observed rows are fixed.
# No current-state publication, automatic model adoption or calibrated peak claim.
suppressPackageStartupMessages(library(mgcv))
suppressPackageStartupMessages(library(nlme))
library(splines)
groups <- c('Control','Learner-flagged','Nonlearner-flagged')
datasets <- list()
for (assay in c('delay','3strace')) {
  d <- read.csv(paste0('/input/',assay,'.csv'), stringsAsFactors=FALSE)
  d$display_group <- factor(d$display_group, levels=groups)
  d$fish_id <- factor(d$fish_id)
  d$time_scaled <- d$time_s/20
  d$bin_index <- as.integer(round((d$time_s+19.75)/.5))
  stopifnot(!anyDuplicated(d[c('fish_id','time_s')]), all(is.finite(d$signed)))
  datasets[[assay]] <- d[order(d$fish_id,d$time_s),]
}
dir.create('/output')
writeLines(capture.output(sessionInfo()),'/output/session-info.txt')

make_fit <- function(d, assay, candidate) {
  random <- if (assay=='delay') list(fish_id=~1) else list(fish_id=~1+time_scaled)
  control <- list(niterEM=0, msMaxIter=200, maxIter=200, msMaxEval=400, returnObject=FALSE)
  if (startsWith(candidate,'LMM')) {
    df <- as.integer(sub('LMM','',candidate))
    form <- as.formula(sprintf('signed ~ display_group * bs(time_s, df=%d, degree=3, Boundary.knots=c(-20,20))',df))
    model <- lme(form, random=random, data=d, method='ML',
                 control=lmeControl(msMaxIter=200,maxIter=200,msMaxEval=400,returnObject=FALSE))
    list(model=model, lme=model, gam=NULL, kind='LMM',df=df)
  } else {
    k <- if(candidate=='GAMM30') 30 else 20
    form <- as.formula(sprintf('signed ~ display_group + s(time_s, by=display_group, bs="cr", k=%d)',k))
    corr <- if (candidate=='GAMM20_AR1') corAR1(form=~bin_index|fish_id) else NULL
    model <- gamm(form,random=random,correlation=corr,data=d,method='REML',control=control)
    list(model=model, lme=model$lme, gam=model$gam,kind='GAMM',df=k)
  }
}

population_predict <- function(fitted, newdata) {
  if (fitted$kind=='LMM') as.numeric(predict(fitted$model,newdata=newdata,level=0))
  else as.numeric(predict(fitted$gam,newdata=newdata,type='response'))
}

run_fit <- function(assay,candidate,fold) {
  all <- datasets[[assay]]
  train <- if(fold==0) all else droplevels(all[all$fold!=fold,])
  test <- if(fold==0) all else all[all$fold==fold,]
  if(fold>0) stopifnot(!any(as.character(test$fish_id)%in%as.character(train$fish_id)))
  warnings <- character()
  started <- proc.time()[[3]]
  fitted <- tryCatch(withCallingHandlers(make_fit(train,assay,candidate),
    warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
    error=function(e)e)
  base <- paste0('/output/',assay,'-',candidate,'-',fold)
  status <- data.frame(assay=assay,candidate=candidate,fold=fold,success=!inherits(fitted,'error'),
      seconds=proc.time()[[3]]-started,train_rows=nrow(train),test_rows=nrow(test),
      train_fish=nlevels(train$fish_id),test_fish=length(unique(test$fish_id)),
      warning=paste(unique(warnings),collapse=' | '),error='',apvar_ok=NA)
  if(inherits(fitted,'error')) {
    status$error <- conditionMessage(fitted)
    write.csv(status,paste0(base,'-status.csv'),row.names=FALSE)
    return(invisible(FALSE))
  }
  predicted <- population_predict(fitted,test)
  stopifnot(length(predicted)==nrow(test),all(is.finite(predicted)))
  out <- test[c('fish_id','display_group','time_s','signed','fold')]
  out$predicted <- predicted
  write.csv(out,paste0(base,'-predictions.csv'),row.names=FALSE)
  status$apvar_ok <- !is.character(fitted$lme$apVar)
  status$loglik <- as.numeric(logLik(fitted$lme))
  status$residual_scale <- fitted$lme$sigma
  if(fold==0) {
    grid <- expand.grid(time_s=seq(-19.75,19.75,.5),display_group=groups,stringsAsFactors=FALSE)
    grid$display_group <- factor(grid$display_group,levels=groups)
    grid$time_scaled <- grid$time_s/20
    grid$bin_index <- as.integer(round((grid$time_s+19.75)/.5))
    grid$fish_id <- factor(rep(levels(train$fish_id)[1],nrow(grid)),levels=levels(train$fish_id))
    grid$fitted <- population_predict(fitted,grid)
    write.csv(grid[c('display_group','time_s','fitted')],paste0(base,'-curves.csv'),row.names=FALSE)
    # These include estimated fish effects; GAMM lme residuals also include
    # penalized smooth coefficients through their random-effect representation.
    rr <- residuals(fitted$lme,type='response')
    rn <- residuals(fitted$lme,type='normalized')
    stopifnot(length(rr)==nrow(train),length(rn)==nrow(train))
    rd <- train[c('fish_id','display_group','time_s','bin_index')]
    rd$residual <- as.numeric(rr); rd$normalized <- as.numeric(rn)
    write.csv(rd,paste0(base,'-residuals.csv'),row.names=FALSE)
    if(!is.null(fitted$gam)) {
      tab <- as.data.frame(summary(fitted$gam)$s.table)
      tab$term <- rownames(tab)
      # Do not export same-data smooth p-values as scientific evidence.
      tab <- tab[c('term','edf','Ref.df')]
      write.csv(tab,paste0(base,'-edf.csv'),row.names=FALSE)
      write.csv(data.frame(smoothing_parameter=fitted$gam$sp),paste0(base,'-smoothing.csv'),row.names=TRUE)
    }
    saveRDS(fitted,paste0(base,'-model.rds'))
    writeLines(capture.output(VarCorr(fitted$lme)),paste0(base,'-variance.txt'))
    if(candidate=='GAMM20_AR1') {
      rho <- coef(fitted$lme$modelStruct$corStruct,unconstrained=FALSE)
      write.csv(data.frame(rho=as.numeric(rho)),paste0(base,'-rho.csv'),row.names=FALSE)
    }
  }
  write.csv(status,paste0(base,'-status.csv'),row.names=FALSE)
  invisible(TRUE)
}
