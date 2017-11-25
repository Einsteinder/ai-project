#dsn<-read.csv("/Users/klepht/Downloads/test1.csv")
#dsn<-dsn[, -which(colMeans(is.na(dsn)) > 0.5)]
dat2 <- read.csv("/Users/klepht/Downloads/test1.csv", header=T, na.strings=c("","NA"," ","n/a"))
dat3 <- dat2[ lapply( dat2, function(x) sum(is.na(x)) / length(x) ) < 0.1 ]
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
dat4 <- lapply(dat3, function(x){
  # check if you have a factor first:
  if(!is.factor(x)) return(x)
  # otherwise include NAs into factor levels and change factor levels:
  x <- factor(x, exclude=NULL)
  levels(x)[is.na(levels(x))] <- Mode(x)
  return(x)
})


label_encoder = function(vec){
  levels = sort(unique(vec))
  function(x){
    match(x, levels)
  }
}
emp_title = dat4$emp_title
title_encoder = label_encoder(emp_title) # create encoder
dat4$emp_title<-title_encoder(dat4$emp_title)

zip_code = dat4$zip_code
zip_encoder = label_encoder(zip_code)
dat4$zip_code = zip_encoder(dat4$zip_code)

earliest_cr_line = dat4$earliest_cr_line
crline_encoder = label_encoder(earliest_cr_line)
dat4$earliest_cr_line = crline_encoder(dat4$earliest_cr_line)



dat4<-as.data.frame(dat4)



m <- model.matrix( ~0+verification_status+term+train+emp_title+grade+sub_grade+home_ownership+
                     loan_status+purpose+title+zip_code+addr_state+initial_list_status+pymnt_plan+
                     application_type+statusF+status+issue_d+last_pymnt_d+last_credit_pull_d+emp_length,
                   data= dat4)
bc3withm<-cbind(dat4,m)
bc3withmd<-within(bc3withm, rm(url,term,id,train,verification_status,pymnt_plan,issue_d,last_credit_pull_d,last_pymnt_d,member_id,emp_title,emp_length,grade,sub_grade,home_ownership,loan_status,purpose,title,zip_code,addr_state,initial_list_status,application_type,statusF,status))
write.csv(bc3withmd, file = "MyData.csv",row.names=FALSE)
rm(list=ls())
