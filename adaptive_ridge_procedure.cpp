#include <RcppArmadillo.h>
#include <Rcpp.h>
#include<Rmath.h>
#include <cmath>
using namespace Rcpp; // <Rcpp.h>
using namespace arma; // <RcppArmadillo.h>
using namespace std; // C++
using namespace R; // <Rmath.h>
// Describe namespaces except "Rcpp"

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline NumericMatrix band_weights(NumericVector w, int diff) {
  int ws = w.size();
  // Compute the entries of the difference matrix
  NumericVector binom(diff + 1);
  for (int i = 0; i <= diff; i ++) {
    binom(i) = Rf_choose(diff, i) * std::pow((-1), i);
  }
  // Compute the limit indices
  NumericMatrix ind_mat(ws + diff, 2);
  for (int ind = 0; ind < ws + diff; ind ++) {
    ind_mat(ind, 0) = ind - diff < 0 ? 0 : ind - diff;
    ind_mat(ind, 1) = ind < ws - 1 ? ind : ws - 1;
  }
  // Main loop
  NumericMatrix result(ws + diff, diff + 1);
  for (int j = 0; j < result.ncol(); j ++) {
    for (int i = 0; i < ws + diff - j; i ++) {
      double temp = 0.;
      for (int k = ind_mat(i + j, 0); k <= ind_mat(i, 1); k ++) {
        temp += binom(i - k) * binom(i + j - k) * w(k);
      }
      result(i, j) = temp;
    }
  }
  return result;
}

//#' Inverse the hessian and multiply it by the score
//#'
//#' @param par The parameter vector
//#' @param XX_band The matrix \eqn{X^T X} where \code{X} is the design matrix. This argument is given
//#' in the form of a band matrix, i.e., successive columns represent superdiagonals.
//#' @param Xy The vector of currently estimated points \eqn{X^T y}, where \eqn{y} is the y-coordinate of the data.
//#' @param pen Positive penalty constant.
//#' @param w Vector of weights. Has to be of length
//#' @param diff The order of the differences of the parameter. Equals \code{degree + 1} in adaptive spline regression.
//#' @return The solution of the linear system: \deqn{(X^T X + pen D^T diag(w) D) ^ {-1} X^T y - par}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline NumericMatrix hessian_solvers(
    NumericVector par, 
    NumericMatrix XX_band, 
    NumericMatrix Xy, 
    double pen, 
    NumericVector w, 
    int diff){
    if(XX_band.ncol() != diff + 1) stop("Error: XX_band must have diff + 1 columns");
    Environment pkg = Environment::namespace_env("bandsolve");
    Function bandsolve = pkg["bandsolve"];
    NumericMatrix penalty = pen * band_weights(w, diff);
    //arma::mat addition = XX_band + penalty;
    arma::mat band = XX_band + penalty;
    arma::mat Xy_ =Rcpp::as<arma::mat>(Xy);
    //NumericMatrix band(penalty.nrow(), penalty.ncol(), addition.begin());
    //NumericMatrix band_solve = bandsolve(band, Xy);
    arma::mat band_solve = arma::pinv(band) * Xy_;
    arma::mat ans=band_solve- as<arma::mat> (par);
    NumericMatrix result(par.length(), 1, ans.begin());
    return result;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline NumericVector wridge_solvers(
    NumericMatrix XX_band, 
    NumericMatrix Xy, 
    double pen,
    NumericVector w,
    NumericVector old_par,
    int degree = 3, // the degree of B-spline
    double tol = 1e-8 // convergence criterion of Newton-Raphson method
){
  double rel_error;
  NumericVector par;
  LogicalVector idx;
  do{
    par = old_par + hessian_solvers(old_par, XX_band, Xy,
                                    pen, w, degree + 1);
    if(max(abs(old_par))!=0){
      LogicalVector idx = old_par != 0;
      rel_error = max(abs(par[idx] - old_par[idx]) / abs(old_par[idx]));
      old_par = par;}
  }while(rel_error < tol);
  return par;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline List lm_fit(Rcpp::NumericMatrix X, Rcpp::NumericVector y) {
  arma::mat X_ = Rcpp::as<arma::mat>(X);
  arma::colvec y_ = Rcpp::as<arma::colvec>(y);
  int n = X_.n_rows, m=X_.n_cols;
  arma::colvec coef = arma::pinv(X_.t()*X_) * X_.t() * y_;    // using generalized inverse
  arma::colvec res  = y_ - X_*coef;           // residuals
  NumericMatrix coefficients(m,1,coef.begin());
  NumericMatrix residuals(n,1, res.begin());
  return List::create(Named("coefficients") = coefficients,
               Named("residuals")  = residuals);
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline NumericVector Nth_diff(
    NumericVector x, 
    int differences // Nth order difference operator
  ){
  NumericVector Nth_diff;
  for(int i=1; i<=differences; i++){
      Nth_diff=diff(x);
      x=Nth_diff;}
  return Nth_diff;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline NumericMatrix diff_X(
  NumericMatrix X,
  int differences // Nth order difference operator
){
  int n=X.nrow(), m=X.ncol();
  for(int j=1; j<=differences; j++){
    NumericMatrix  Delta(n-j,m);
    for(int i=0; i<m-j; i++){
      NumericVector X_row=X(i+1,_), X_row_=X(i,_);
      Delta(i, _)= X_row - X_row_;
    }
    X=Delta;
  }
  return X;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline NumericVector successive_push_back(NumericVector X1, NumericVector X2){
  int n=X2.size();
  for(int i=0; i<n; i++){
    X1.push_back(X2[i]);
  }
  return X1;
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
inline List adaptive_ridge_procedure (
    NumericVector x, 
    NumericVector y,
    NumericVector knots,
    NumericVector pen,
    int degree = 3,
    double epsilon = 1e-5,
    double tol = 1e-6) {
  Environment spline2 = Environment::namespace_env("splines2");
  Function bSpline = spline2["bSpline"];
  //Environment bandsolve = Environment::namespace_env("bandsolve");
  //Function mat2rot = bandsolve["mat2rot"];
  NumericMatrix X = bSpline(x, Named("knots")=knots, Named("intercept", true), Named("degree")=degree);
  arma::mat X_ = Rcpp::as<arma::mat>(X);
  arma::mat X_t = Rcpp::as<arma::mat>(transpose(X));
  arma::mat XX_ = X_t * X_;
  R_xlen_t n = X.ncol();
  //NumericMatrix XX(n, n, XX_.begin());
  //mat XX__ = XX_ + 1e-20 * arma::eye(n, n);
  //NumericMatrix XX_band_base(n, n, XX__.begin());
  //NumericMatrix XX_band_base_ = mat2rot(XX_band_base);
  //NumericVector add(n,0);
  //NumericMatrix XX_band = cbind(XX_band_base_, add);
  arma::vec y_ = Rcpp::as<arma::vec>(y);
  arma::mat Xy_ = X_t * y_;
  //NumericMatrix Xy(X.ncol(), 1,Xy_.begin());
  //Define returned values
  R_xlen_t N = pen.length();
  List knots_sel(N); //X_sel(N), par_ls(N), sel_ls(N), 
  NumericVector aic(N), bic(N), ebic(N), dim(N), loglik(N);
  //Initialize values
  R_xlen_t nn = X.ncol() - degree - 1;
  NumericVector par_(n,1), w(nn,1), old_sel(nn, 1), ep(n,epsilon*epsilon);
  arma::mat I = arma::eye(n,n);
  NumericMatrix In(n, n, I.begin());
  arma::mat D = as<arma::mat> (diff_X(In, degree+1));
  //Main loop
  for(int ind_pen=0; ind_pen<N; ind_pen++){
    //NumericVector par = wridge_solvers(XX_band, Xy, pen[ind_pen], w, par_);
    //par_ = par;
    //w=pow(pow(Nth_diff(par, degree + 1),2) + ep, -1);
    //NumericVector sel =w*pow(Nth_diff(par, degree + 1),2);
    //LogicalVector dif_idx = old_sel-sel != NA_LOGICAL;
    //LogicalVector convergence = max(abs(old_sel[dif_idx]-sel[dif_idx])) < tol;
    bool convergence;
    NumericVector sel;
    do{
      arma::mat W=arma::diagmat(as<arma::colvec>(w));
      colvec par_ = arma::pinv(XX_+double(pen[ind_pen])* D.t()*W*D)*Xy_;
      NumericVector par(par_.begin(),par_.end());
      w=pow(pow(Nth_diff(par, degree + 1),2) + ep, -1);
      sel =w*pow(Nth_diff(par, degree + 1),2);
      LogicalVector dif_idx = old_sel-sel != NA_LOGICAL;
      convergence = max(abs(old_sel[dif_idx]-sel[dif_idx])) < tol;
      old_sel = sel;
    }while(!convergence);
    //results
    //sel_ls[ind_pen] = sel;
    knots_sel[ind_pen] = knots[sel > 0.99];
    NumericMatrix design = bSpline(x, Named("knots")=knots_sel[ind_pen], Named("intercept", true), Named("degree")=degree);
    //X_sel[ind_pen] = design;
    NumericVector sels = successive_push_back(sel, rep(double(1), degree+1));
    LogicalVector idx = sels > 0.99;
    //NumericVector par_ls_ = rep(double(0), n);
    List model = lm_fit(design,y);
    //par_ls_[idx] = as<NumericVector> (model["coefficients"]);
    //par_ls[ind_pen]=clone(par_ls_);
    loglik[ind_pen] =  -design.nrow()*std::log(float(2*arma::datum::pi))/2 - design.nrow()/2 -
      design.nrow()*std::log(float(mean(pow(as<NumericVector> (model["residuals"]), 2))))/2;
    dim[ind_pen] = (as<NumericVector>(knots_sel[ind_pen])).length() + degree + 1;
    aic[ind_pen] = -2*loglik[ind_pen]+2*dim[ind_pen];
    bic[ind_pen] = -2*loglik[ind_pen] + std::log(X.nrow()) * dim[ind_pen];
    ebic[ind_pen] =  bic[ind_pen] + 
      2 * R::lchoose(X.ncol(), design.ncol());
    }
  R_xlen_t n_aic = which_min(aic);
  R_xlen_t n_bic = which_min(bic);
  R_xlen_t n_ebic = which_min(ebic);
  
  //return List::create(Named("sel_ls")=sel_ls);
  return List::create(
    Named("AIC")=aic[n_aic], Named("BIC")=bic[n_bic],Named("EBIC")=ebic[n_ebic], 
          Named("knots_AIC")=knots_sel[n_aic],Named("knots_BIC")=knots_sel[n_bic],Named("knots_EBIC")=knots_sel[n_ebic]);
}