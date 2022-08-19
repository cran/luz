## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  eval = FALSE
)

## ----setup--------------------------------------------------------------------
#  library(torch)
#  library(luz)

## -----------------------------------------------------------------------------
#  net <- nn_module(
#    "Net",
#    initialize = function() {
#      self$fc1 <- nn_linear(100, 50)
#      self$fc1 <- nn_linear(50, 10)
#    },
#    forward = function(x) {
#      x %>%
#        self$fc1() %>%
#        nnf_relu() %>%
#        self$fc2()
#    }
#  )

## -----------------------------------------------------------------------------
#  fitted <- net %>%
#    setup(
#      loss = nn_cross_entropy_loss(),
#      optimizer = optim_adam,
#      metrics = list(
#        luz_metric_accuracy
#      )
#    ) %>%
#    fit(train_dl, epochs = 10, valid_data = test_dl)

## -----------------------------------------------------------------------------
#  net <- nn_module(
#    "Net",
#    initialize = function() {
#      self$fc1 <- nn_linear(100, 50)
#      self$fc1 <- nn_linear(50, 10)
#    },
#    forward = function(x) {
#      x %>%
#        self$fc1() %>%
#        nnf_relu() %>%
#        self$fc2()
#    },
#    set_optimizers = function(lr_fc1 = 0.1, lr_fc2 = 0.01) {
#      list(
#        opt_fc1 = optim_adam(self$fc1$parameters, lr = lr_fc1),
#        opt_fc2 = optim_adam(self$fc2$parameters, lr = lr_fc2)
#      )
#    },
#    loss = function(input, target) {
#      pred <- ctx$model(input)
#  
#      if (ctx$opt_name == "opt_fc1")
#        nnf_cross_entropy(pred, target) + torch_norm(self$fc1$weight, p = 1)
#      else if (ctx$opt_name == "opt_fc2")
#        nnf_cross_entropy(pred, target)
#    }
#  )

## -----------------------------------------------------------------------------
#  fitted <- net %>%
#    setup(metrics = list(luz_metric_accuracy)) %>%
#    fit(train_dl, epochs = 10, valid_data = test_dl)

## -----------------------------------------------------------------------------
#  net <- nn_module(
#    "Net",
#    initialize = function() {
#      self$fc1 <- nn_linear(100, 50)
#      self$fc1 <- nn_linear(50, 10)
#    },
#    forward = function(x) {
#      x %>%
#        self$fc1() %>%
#        nnf_relu() %>%
#        self$fc2()
#    },
#    set_optimizers = function(lr_fc1 = 0.1, lr_fc2 = 0.01) {
#      list(
#        opt_fc1 = optim_adam(self$fc1$parameters, lr = lr_fc1),
#        opt_fc2 = optim_adam(self$fc2$parameters, lr = lr_fc2)
#      )
#    },
#    step = function() {
#      ctx$loss <- list()
#      for (opt_name in names(ctx$optimizers)) {
#  
#        pred <- ctx$model(ctx$input)
#        opt <- ctx$optimizers[[opt_name]]
#        loss <- nnf_cross_entropy(pred, target)
#  
#        if (opt_name == "opt_fc1") {
#          # we have L1 regularization in layer 1
#          loss <- nnf_cross_entropy(pred, target) +
#            torch_norm(self$fc1$weight, p = 1)
#        }
#  
#        if (ctx$training) {
#          opt$zero_grad()
#          loss$backward()
#          opt$step()
#        }
#  
#        ctx$loss[[opt_name]] <- loss$detach()
#      }
#    }
#  )

