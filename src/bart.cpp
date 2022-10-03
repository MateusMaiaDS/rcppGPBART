#define _USE_MATH_DEFINES
#include<cmath>
#include <math.h>
#include<Rcpp.h>
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
                sum_r = sum_r + residuals_values(residuals_values(i));
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
                int node_min_size,const Rcpp::NumericMatrix& xcut,
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
                Rcpp::NumericMatrix all_node_index;

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
                arma::vec x_current_node(100) ;

                for(int i = 0; i<g_node->obs_train.size();i++) {
                        x_current_node(i) = (x_train(g_node->obs_train(i),split_var));
                }

                min_x_current_node = min(x_current_node);
                max_x_current_node = max(x_current_node);

                // Getting available xcut variables
                x_cut_candidates = xcut(_,split_var);

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
                                cout << "RETURNING BEFORE" << endl;
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
                curr_res.push_back(g_node -> obs_train(j));

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

        cout << "ACCEPTANCE IS: " << exp(log_acceptance) << endl;
        // Updating the tree
        if(R::runif(0,1)<=exp(log_acceptance)){

                cout << " TREE ACCEPTED" << endl;
                // Updating the current_tree
                list_node.insert(list_node.begin()+g_original_index+1,
                                 new_left_node);

                list_node.insert(list_node.begin()+g_original_index+2,
                                 new_right_node);
        }
        // Finishing the process;
        return;

};



//[[Rcpp::export]]
void test_grow_tree(arma::mat x_train,
               arma::mat x_test,
               arma:: vec y_train,
               arma:: mat x_cut,
               double alpha,
               double beta,
               int node_min_size,

               double tau,
               double tau_mu){

        Tree new_tree(x_train.n_rows, x_test.n_rows);
        new_tree.DisplayNodes();
        new_tree.grow(y_train,x_train,x_test,node_min_size,x_cut,tau,tau_mu,alpha,beta);
        new_tree.DisplayNodes();

        return;

}

