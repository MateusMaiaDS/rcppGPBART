#define _USE_MATH_DEFINES
#include<cmath>
#include <math.h>
#include<RcppArmadillo.h>
#include <vector>
#include "tree.h"

using namespace std;

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins(cpp11)]]
// The line above (depends) it will make all the depe ndcies be included on the file
using namespace Rcpp;

// Loglikelihood of a node
double loglikelihood(Rcpp::NumericVector& residuals_values,double tau,double tau_mu){

        // Declaring quantities in the node
        int n_train = residuals_values.size();
        double sum_r_sq = 0;
        double sum_r = 0;

        for(int i = 0; i<n_train;i++){
                sum_r_sq =  sum_r_sq + residuals_values(i)*residuals_values(i);
                sum_r = sum_r + residuals_values(i);
        }

        return -0.5*tau*sum_r_sq -0.5*log(tau_mu+(n_train*tau)) + (0.5*(tau*tau)*(sum_r*sum_r))/( (tau*n_train)+tau_mu );
};


// Updating nu in a node;
void Node::update_mu(arma::vec& residuals_values,double tau,double tau_mu){
        // Calculating the sum of residuals
        double sum_r = 0;
        double n_train = obs_train.size();

        for(int i = 0; i<obs_train.size();i++){
                sum_r+=residuals_values(obs_train(i));
        }


        mu = R::rnorm((sum_r*tau)/(tau*n_train+tau_mu),1/sqrt(n_train*tau+tau_mu));

};


// Update tree all the mu from all the trees
void Tree::update_mu_tree(arma::vec & res_val, double tau, double tau_mu){

        // Iterating over all nodes
        for(int i = 0; i<list_node.size();i++){
                if(list_node[i].isTerminal()==1){
                        list_node[i].update_mu(res_val,tau,tau_mu);
                }
        }

}



// Growing a tree
void Tree::grow(arma::vec res_values,
                const arma::mat& x_train,const arma::mat& x_test,
                int node_min_size,const Rcpp::NumericMatrix& x_cut,
                double& tau, double& tau_mu, double& alpha, double& beta){

        // Defining the number of covariates
        int p = x_train.n_cols;
        int cov_trial_counter = 0; // To count if all the covariate were tested
        int valid_split_indicator = 0;
        int n_train,n_test;
        int g_original_index;
        int split_var;
        int max_index;
        int g_index;
        double min_x_current_node;
        double max_x_current_node;


        Rcpp::NumericVector x_cut_valid; // Defining the vector of valid "split rules" based on xcut and the terminal node
        Rcpp::NumericVector x_cut_candidates;
        Node* g_node; // Node to be grow
        int nog_counter;
        // Select a terminal node to be randomly selected to split
        vector<Node> candidate_nodes = getTerminals();
        int n_t_node ;

        // Find a valid split indicator within a valid node
        while(valid_split_indicator==0){

                // Getting the nog counter
                nog_counter = 0;

                // Getting the number of terminal nodes
                n_t_node = candidate_nodes.size();

                // Getting a NumericVector of all node index
                Rcpp::NumericVector all_node_index;

                // Selecting a random terminal node
                g_index = sample_int(candidate_nodes.size());

                // Iterating over all nodes

                for(int i=0;i<list_node.size();i++){

                        all_node_index.push_back(list_node[i].index); // Adding the index into the list

                        if(list_node[i].index==candidate_nodes[g_index].index){
                                g_original_index = i;
                                nog_counter++;
                        }

                        // Getting nog (PROPER ONE FOR THE TRANSITION)
                        if(list_node[i].left != -1 & list_node[i].right != -1){
                                if(list_node[list_node[i].left].left == -1 & list_node[list_node[i].left].right == -1){
                                        if(list_node[list_node[i].right].left == -1 & list_node[list_node[i].right].right == -1){
                                                if(list_node[i].depth==candidate_nodes[g_index].depth){
                                                        nog_counter++;
                                                };
                                        };
                                };
                        };
                };


                // Getting the maximum index in case to build the new indexes for the new terminal nodes
                max_index = max(all_node_index);

                // Selecting the terminal node
                g_node = &candidate_nodes[g_index];

                // Selecting the random rule
                split_var = sample_int(p);

                // Getting the number of train
                n_train = g_node->obs_train.size();
                n_test = g_node->obs_test.size();

                // Selecting the column of the x_curr
                Rcpp::NumericVector x_current_node ;

                for(int i = 0; i<g_node->obs_train.size();i++) {
                        x_current_node.push_back(x_train(g_node->obs_train(i),split_var));
                }

                min_x_current_node = min(x_current_node);
                max_x_current_node = max(x_current_node);

                // Getting available xcut variables
                x_cut_candidates = x_cut(_,split_var);

                // Create a vector of splits that will lead to nontrivial terminal nodes
                for(int k = 0; k<x_cut_candidates.size();k++){
                        if(x_cut_candidates(k)>min_x_current_node && x_cut_candidates(k)<max_x_current_node){
                                x_cut_valid.push_back(x_cut_candidates(k));
                        }
                }

                // Do not grow a tree and skip it or select a new tree
                if(x_cut_valid.size()==0){

                        cov_trial_counter++;
                        // CHOOSE ANOTHER SPLITING RULE - if reach the limit p do not do anything
                        if(cov_trial_counter == p){
                                return;
                        }

                } else {

                        // Verifying that a valid split was selected
                        valid_split_indicator = 1;

                }
        }

        // Sample a g_var
        double split_var_rule = x_cut_valid(sample_int(x_cut_valid.size()));

        // Creating the new terminal nodes based on the new xcut selected value
        // Getting observations that are on the left and the ones that are in the right
        Rcpp::NumericVector new_left_train_index;
        Rcpp::NumericVector new_right_train_index;
        Rcpp::NumericVector resid_left_train;
        Rcpp::NumericVector resid_right_train;
        Rcpp::NumericVector curr_obs_train; // Observations that belong to that terminal node

        // Same from above but for test observations
        // Getting observations that are on the left and the ones that are in the right
        Rcpp::NumericVector new_left_test_index;
        Rcpp::NumericVector new_right_test_index;
        Rcpp::NumericVector resid_left_test;
        Rcpp::NumericVector resid_right_test;
        Rcpp::NumericVector curr_obs_test; // Observations that belong to that terminal node

        // Get the current vector of residuals
        Rcpp::NumericVector curr_res;

        /// Iterating over the train and test
        for(int j=0; j<n_train;j++){

                // Getting the current node of residuals
                curr_res.push_back(res_values[g_node -> obs_train(j)]);

                if(x_train(g_node->obs_train(j),split_var)<=split_var_rule){
                        new_left_train_index.push_back(g_node->obs_train(j));
                        // Creating the residuals list
                        resid_left_train.push_back(res_values(g_node->obs_train(j)));

                } else {
                        new_right_train_index.push_back(g_node->obs_train(j));
                        resid_right_train.push_back(res_values(g_node->obs_train(j)));
                }
        }

        /// Iterating over the test and test
        for(int i=0; i<n_test;i++){
                if(x_test(g_node->obs_test(i),split_var)<=split_var_rule){
                        // Saving the index
                        new_left_test_index.push_back(g_node->obs_test(i));
                } else {
                        new_right_test_index.push_back(g_node->obs_test(i));

                }
        }


        // Updating the current node
        list_node[g_original_index].left = max_index+1;
        list_node[g_original_index].right = max_index+2;

        Node new_left_node = Node(max_index+1,
                                  new_left_train_index,
                                  new_left_test_index,
                                  -1,
                                  -1,
                                  g_node->depth+1,
                                  split_var,
                                  split_var_rule,
                                  0);

        Node new_right_node = Node(max_index+2,
                                  new_right_train_index,
                                  new_right_test_index,
                                  -1,
                                  -1,
                                  g_node->depth+1,
                                  split_var,
                                  split_var_rule,
                                  0);


        // Calculating all loglikelihood
        double log_acceptance = loglikelihood(resid_left_train,tau,tau_mu) + loglikelihood(resid_right_train,tau,tau_mu) - loglikelihood(curr_res,tau,tau_mu)+ // Node loglikelihood
                log(0.3/nog_counter)-log(0.3/candidate_nodes.size())+ // Transition prob
                2*log(1-alpha/pow(1+(list_node[g_original_index].depth+1),beta))+beta*log(alpha*(1+list_node[g_original_index].depth))-log(1-alpha/pow(1+list_node[g_original_index].depth,beta)); // Tree prior

        // Updating the tree
        if(R::runif(0,1)<=exp(log_acceptance)){

                // Updating the current_tree
                list_node.insert(list_node.begin()+g_original_index+1,
                                 new_left_node);

                list_node.insert(list_node.begin()+g_original_index+2,
                                 new_right_node);
        }
        // Finishing the process;
        return;

};

// Pruning a tree
void Tree::prune(arma::vec& res_val,double tau, double tau_mu, double& alpha, double& beta){

        // Selecting possible parents of terminal nodes to prune
        vector<Node> nog_list;
        Rcpp::NumericVector nog_original_index_vec;
        int nog_counter = 0;
        int n_terminal = 0;
        int id_node;
        // Do not prune a ROOT never
        if(list_node.size()==0){
                return;
        }

        // Getting the parent of terminal node only
        for(int i = 0; i<list_node.size();i++){
                if(list_node[i].isTerminal()==0){
                        if(list_node[i+1].isTerminal()== 1 && list_node[i+2].isTerminal()==1){
                                nog_list.push_back(list_node[i]);
                                nog_original_index_vec.push_back(i);
                                nog_counter++;
                        }
                } else {
                        n_terminal++;
                }
        }

        // Sampling a random node to be pruned
        int p_node_index = sample_int(nog_list.size());
        int nog_original_index = nog_original_index_vec(p_node_index);
        Node p_node = nog_list[p_node_index];

        // Identifying which node was pruned
        id_node = nog_original_index;

        // IN CASE OF ACCEPTING
        Rcpp::NumericVector res_left_val;
        Rcpp::NumericVector res_right_val;
        Rcpp::NumericVector res_curr_val;

        // Creating the left residuals vector
        for(int l = 0;l<list_node[id_node+1].obs_train.size();l++){
                res_left_val.push_back(res_val(list_node[id_node+1].obs_train(l)));
        }

        // Creating the right residuals vector
        for(int r = 0; r <list_node[id_node+2].obs_train.size(); r++){
                res_right_val.push_back(res_val(list_node[id_node+2].obs_train(r)));
        }

        // Creating the current residuals vector
        for(int c = 0; c < list_node[id_node].obs_train.size(); c++){
                res_curr_val.push_back(res_val(list_node[id_node].obs_train(c)));
        }

        double log_acceptance = loglikelihood(res_curr_val,tau,tau_mu) - loglikelihood(res_left_val,tau,tau_mu) - loglikelihood(res_right_val,tau,tau_mu)+ // Node loglikelihood
                log(0.3/(n_terminal-1))-log(0.3/nog_counter) + // Transition prob
                log(1-alpha/pow(1+list_node[id_node].depth,beta))-(2*log(1-alpha/pow(1+(list_node[id_node].depth+1),beta))+beta*log(alpha*(1+list_node[id_node].depth))); // Tree prior

        // Pruning the tree
        if(R::runif(0,1) < exp(log_acceptance)){
                list_node[nog_original_index].left = -1;
                list_node[nog_original_index].right = -1;

                // Erasing the new vectors
                list_node.erase(list_node.begin()+id_node+1,list_node.begin()+id_node+3);

        }

        return;

}



// Growing a tree
void Tree::change(arma::vec res_values,
                const arma::mat& x_train,const arma::mat& x_test,
                int node_min_size,const Rcpp::NumericMatrix& x_cut,
                double& tau, double& tau_mu, double& alpha, double& beta){

        // Selecting possible parents of terminal nodes to prune
        vector<Node> nog_list;
        Rcpp::NumericVector nog_original_index_vec;

        // Getting the parent of terminal node only
        for(int i = 0; i<list_node.size();i++){
                if(list_node[i].isTerminal()==0){
                        if(list_node[i+1].isTerminal()== 1 && list_node[i+2].isTerminal()==1){
                                nog_list.push_back(list_node[i]);
                                nog_original_index_vec.push_back(i);
                        }
                }
        }

        // Sampling a random node to be changed

        Node* c_node; // Node to be grow
        int p_node_index = sample_int(nog_list.size());
        int nog_original_index = nog_original_index_vec(p_node_index);
        c_node = &nog_list[p_node_index];


        // cout << "NODE CHANGED IS " << c_node->index << endl;

        // Defining the number of covariates
        int p = x_train.n_cols;
        int cov_trial_counter = 0; // To count if all the covariate were tested
        int valid_split_indicator = 0;
        int split_var;
        double min_x_current_node;
        double max_x_current_node;

        Rcpp::NumericVector x_cut_valid; // Defining the vector of valid "split rules" based on xcut and the terminal node
        Rcpp::NumericVector x_cut_candidates;

        // Find a valid split indicator within a valid node
        while(valid_split_indicator == 0){

                // Getting the split var
                split_var = sample_int(p);

                // cout << "ERROR 1 " << endl;
                // Selecting the x_curr
                Rcpp::NumericVector x_current_node ;
                for(int i = 0; i<c_node->obs_train.size();i++) {
                        x_current_node.push_back(x_train(c_node->obs_train(i),split_var));
                }

                min_x_current_node = min(x_current_node);
                max_x_current_node = max(x_current_node);

                // Getting available xcut variables
                x_cut_candidates = x_cut(_,split_var);

                // Create a vector of splits that will lead to nontrivial terminal nodes
                for(int k = 0; k<x_cut_candidates.size();k++){
                        if(x_cut_candidates(k)>min_x_current_node && x_cut_candidates(k)<max_x_current_node){
                                x_cut_valid.push_back(x_cut_candidates(k));
                        }
                }


                // IN THIS CASE I GUESS WE WILL ALMOST NEVER GONNA GET HERE SINCE IT'S ALREADY A VALID SPLIT FROM
                //A GROW MOVE
                if(x_cut_valid.size()==0){

                        cov_trial_counter++;
                        // CHOOSE ANOTHER SPLITING RULE
                        if(cov_trial_counter == p){
                                return;
                        }
                } else {

                        // Verifying that a valid split was selected
                        valid_split_indicator = 1;
                }

        }



        // Sample a g_var
        double split_var_rule = x_cut_valid(sample_int(x_cut_valid.size()));

        // Creating the new terminal nodes based on the new xcut selected value
        // Getting observations that are on the left and the ones that are in the right
        Rcpp::NumericVector new_left_train_index;
        Rcpp::NumericVector new_right_train_index;
        Rcpp::NumericVector new_resid_left_train;
        Rcpp::NumericVector new_resid_right_train;
        Rcpp::NumericVector curr_resid_left_train;
        Rcpp::NumericVector curr_resid_right_train;
        Rcpp::NumericVector curr_obs_train; // Observations that belong to that terminal node

        // Same from above but for test observations
        // Getting observations that are on the left and the ones that are in the right
        Rcpp::NumericVector new_left_test_index;
        Rcpp::NumericVector new_right_test_index;
        Rcpp::NumericVector resid_left_test;
        Rcpp::NumericVector resid_right_test;
        Rcpp::NumericVector curr_obs_test; // Observations that belong to that terminal node

        // Get the current vector of residuals
        Rcpp::NumericVector curr_res;

        /// Iterating over the train and test
        for(int j=0; j<list_node[nog_original_index].obs_train.size();j++){

                // UPDATING FOR THE CURRENT NODES
                // cout << "ERROR 2 " << endl;

                if(x_train(c_node->obs_train(j),list_node[nog_original_index+1].var)<=list_node[nog_original_index+1].var_split){
                        curr_resid_left_train.push_back(res_values(c_node->obs_train(j)));
                } else {
                        curr_resid_right_train.push_back(res_values(c_node->obs_train(j)));
                }

                // cout << "ERROR 3 " << endl;

                // Updating for the NEW CHANGED NODES
                if(x_train(c_node->obs_train(j),split_var)<=split_var_rule){
                        new_left_train_index.push_back(c_node->obs_train(j));
                        // Creating the residuals list
                        new_resid_left_train.push_back(res_values(c_node->obs_train(j)));
                } else {
                        new_right_train_index.push_back(c_node->obs_train(j));
                        new_resid_right_train.push_back(res_values(c_node->obs_train(j)));
                }
                // cout << "ERROR 4 " << endl;


        }


        // cout << "ERROR 5 " << endl;

        /// Iterating over the test and test
        for(int i=0; i<list_node[nog_original_index].obs_test.size();i++){
                if(x_test(c_node->obs_test(i),split_var)<=split_var_rule){
                        // Saving the index
                        new_left_test_index.push_back(c_node->obs_test(i));
                } else {
                        new_right_test_index.push_back(c_node->obs_test(i));

                }
        }

        // cout << "ERROR 6 " << endl;


        // Calculating all loglikelihood
        double log_acceptance = loglikelihood(new_resid_left_train,tau,tau_mu) + loglikelihood(new_resid_right_train,tau,tau_mu) - loglikelihood(curr_resid_left_train,tau,tau_mu) - loglikelihood(curr_resid_right_train,tau,tau_mu);

        // Updating the tree
        if(R::runif(0,1)<=exp(log_acceptance)){

                // TREE CHANGED!
                // cout << "TREE " << list_node[nog_original_index].index << " CHANGED" <<endl;
               // Updating all the left and right nodes, respectively
               list_node[nog_original_index+1].obs_train =  new_left_train_index;
               list_node[nog_original_index+1].obs_test = new_left_test_index;
               list_node[nog_original_index+1].var = split_var;
               list_node[nog_original_index+1].var_split = split_var_rule;

               list_node[nog_original_index+2].obs_train =  new_right_train_index;
               list_node[nog_original_index+2].obs_test = new_right_test_index;
               list_node[nog_original_index+2].var = split_var;
               list_node[nog_original_index+2].var_split = split_var_rule;

        }
        // Finishing the process;
        return;

};


// Updating tree prediction
void Tree::getPrediction(arma::vec &train_pred_vec,
                         arma::vec &test_pred_vec){

        // Iterating over all trees nodes
        for(int i=0; i<list_node.size();i++){

                // Checking only over terminal nodes
                if(list_node[i].isTerminal()==1){

                        // Iterating over the train observations
                        for(int j=0;j<list_node[i].obs_train.size();j++){
                                train_pred_vec(list_node[i].obs_train(j)) = list_node[i].mu;
                        }

                        // Iterating over the test observations
                        for(int k = 0; k<list_node[i].obs_test.size();k++){
                                test_pred_vec(list_node[i].obs_test(k)) = list_node[i].mu;
                        }

                }

        } // Finishing the interests over the terminal nodes

        return;
}


double update_tau(const arma::vec& y,
                  arma::vec& y_hat,
                  double a_tau,
                  double d_tau){

        // Function used in the development of the package where I checked
        // contain_nan(y_hat);
        int n = y.size();
        double sum_sq_res = 0;
        for(int i = 0;i<n;i++){
                sum_sq_res = sum_sq_res + (y[i]-y_hat[i])*(y[i]-y_hat[i]);
        }
        return R::rgamma((0.5*n+a_tau),1/(0.5*sum_sq_res+d_tau));
}


//[[Rcpp::export]]
List bart(const arma::mat& x_train,
          const arma::vec& y,
          const arma::mat& x_test,
          const Rcpp::NumericMatrix& x_cut,
          int n_tree,
          int n_mcmc,
          int n_burn,
          int n_min_size,
          double tau, double mu,
          double tau_mu,
          double alpha, double beta,
          double a_tau, double d_tau) {

        // Declaring common variables
        int post_counter = 0;
        double verb;

        // Creating the variables
        int n_train = x_train.n_rows;
        int n_test = x_test.n_rows;
        int n_post = (n_mcmc-n_burn);
        arma::mat y_train_hat_post(n_train,n_post);
        arma::mat y_test_hat_post(n_test,n_post);

        // Getting tau posterior
        arma::vec tau_post(n_post);


        // Getting the initial tree
        Tree init_tree(n_train,n_test);

        // Creating the list of trees
        vector<Tree> current_trees;
        for(int i = 0;i<n_tree;i++){
                current_trees.push_back(init_tree);
        }

        // Creating a matrix of zeros of y_hat
        y_train_hat_post.zeros();
        y_test_hat_post.zeros();

        // Creating the partial residuals and partial predictions
        arma::vec partial_pred(n_train), partial_residuals(n_train);
        arma::vec prediction_train(n_train), prediction_test(n_test); // Creating the vector only for predictions
        arma::vec prediction_test_sum(n_test);

        // Initializing the zero values
        partial_pred.zeros();
        partial_residuals.zeros();
        prediction_train.zeros();
        prediction_test.zeros();
        arma::mat tree_fits_store(n_train,n_tree);
        tree_fits_store.zeros();

        // Iterating over the MCMC samples
        for(int i = 0; i<n_mcmc; i++) {


                prediction_test_sum.zeros();



                for(int t = 0; t<n_tree;t++){

                        // Updating the partial residuals
                        partial_residuals = y - partial_pred + tree_fits_store.col(t);

                        //Setting probabilities to the choice of the verb
                        // Grow: 0-0.3;
                        // Prune: 0.3-0.6
                        // Change: 0.6-1.0
                        // Swap: Not in this current implementation
                        verb = R::runif(0,1);


                        // Forcing stumps to grow
                        if(current_trees[t].list_node.size()==1){
                                verb = 0.1;
                        }


                        // Choosing the verb
                        if( verb < 0.3){
                                // cout << " GROW THE TREE " << endl;
                                current_trees[t].grow(partial_residuals,x_train,x_test,n_min_size,x_cut,tau,tau_mu,alpha,beta);
                        } else if(verb >=0.3 & verb<0.6){
                                // cout << " PRUNE THE TREE " << endl;
                                current_trees[t].prune(partial_residuals,tau,tau_mu,alpha,beta);
                        } else {
                                // cout << " CHANGE THE TREE" << endl;
                                current_trees[t].change(partial_residuals,x_train,x_test,n_min_size,x_cut,tau,tau_mu,alpha,beta);
                        }


                        // cout << " VERB SUCCESSS!!! " << endl;
                        // cout << "Iter number "<< i << endl;

                        // Updating the mu parameters
                        current_trees[t].update_mu_tree(partial_residuals,tau,tau_mu);
                        current_trees[t].getPrediction(prediction_train,prediction_test);

                        // Replacing the value for the partial pred
                        partial_pred = partial_pred-tree_fits_store.col(t) + prediction_train;
                        tree_fits_store.col(t) = prediction_train;

                        // Summing up the test prediction
                        prediction_test_sum += prediction_test;
                };

                // Updating tau
                tau = update_tau(y,partial_pred,tau,tau_mu);

                // cout << "Iter number " << i << endl;

                if(i >= n_burn){
                        y_train_hat_post.col(post_counter) = partial_pred;
                        y_test_hat_post.col(post_counter) = prediction_test_sum;

                        // Updating tau
                        tau_post(post_counter) = tau;
                        post_counter++;
                }
        }

        return Rcpp::List::create(_["y_train_hat_post"] = y_train_hat_post,
                                  _["y_test_hat_post"] = y_test_hat_post,
                                  _["tau_post"] = tau_post);

}


//[[Rcpp::export]]
void test_grow_tree(arma::mat x_train,
               arma::mat x_test,
               arma:: vec y_train,
               Rcpp::NumericMatrix x_cut,
               double alpha,
               double beta,
               int node_min_size,
               double tau,
               double tau_mu){

        Tree new_tree(x_train.n_rows, x_test.n_rows);
        new_tree.DisplayNodes();
        for(int i = 0;i<10;i++){
                new_tree.grow(y_train,x_train,x_test,node_min_size,x_cut,tau,tau_mu,alpha,beta);
        }
        new_tree.DisplayNodes();

        return;

}

//[[Rcpp::export]]
void test_prune_tree(arma::mat x_train,
                    arma::mat x_test,
                    arma:: vec y_train,
                    Rcpp::NumericMatrix x_cut,
                    double alpha,
                    double beta,
                    int node_min_size,

                    double tau,
                    double tau_mu){

        Tree new_tree(x_train.n_rows, x_test.n_rows);
        // new_tree.DisplayNodes();
        for(int i = 0;i<10;i++){
                new_tree.grow(y_train,x_train,x_test,node_min_size,x_cut,tau,tau_mu,alpha,beta);
        }
        cout << " Current tree size " << new_tree.list_node.size() << endl;
        new_tree.prune(y_train,tau,tau_mu,alpha,beta);
        cout << "Prune One Treee size: " << new_tree.list_node.size() << endl;
        new_tree.DisplayNodes();
        new_tree.prune(y_train,tau,tau_mu,alpha,beta);
        cout << "Prune Two Treee size: " << new_tree.list_node.size() << endl;
        new_tree.DisplayNodes();
        return;

}

//[[Rcpp::export]]
void test_change_tree(arma::mat x_train,
                     arma::mat x_test,
                     arma:: vec y_train,
                     Rcpp::NumericMatrix x_cut,
                     double alpha,
                     double beta,
                     int node_min_size,

                     double tau,
                     double tau_mu){

        Tree new_tree(x_train.n_rows, x_test.n_rows);
        // new_tree.DisplayNodes();
        for(int i = 0;i<2;i++){
                new_tree.grow(y_train,x_train,x_test,node_min_size,x_cut,tau,tau_mu,alpha,beta);
        }
        cout << " Current tree size " << new_tree.list_node.size() << endl;
        new_tree.change(y_train,x_train,x_test,node_min_size,x_cut,tau,tau_mu,alpha,beta);
        cout << "Change One Treee size: " << new_tree.list_node.size() << endl;
        new_tree.DisplayNodes();
        new_tree.change(y_train,x_train,x_test,node_min_size,x_cut,tau,tau_mu,alpha,beta);
        cout << "Change Two Treee size: " << new_tree.list_node.size() << endl;
        new_tree.DisplayNodes();
        return;

}

//[[Rcpp::export]]
arma::mat matrix_leak(arma::mat x_train){
        return x_train;
}

