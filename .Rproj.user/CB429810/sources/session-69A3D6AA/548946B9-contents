#define _USE_MATH_DEFINES
#include<cmath>
#include<iostream>
#include<armadillo>
#include <cstdlib>

#include<vector>

#include<Rcpp.h>
using namespace std;


// ====
// Defining minor but auxiliar functions
int sample_int(int n){
        return rand() % n;
}

//Half-cauchy density
double dhalf_cauchy(double x, double mu, double sigma){

        if(x>=mu){
                return (1/(M_PI_2*sigma))*(1/(1+((x-mu)*(x-mu))/(sigma*sigma)));
        } else {
                return 0.0;
        }
}

// Creating one sequence of values
Rcpp::NumericVector seq_along_cpp(int n){
        Rcpp::NumericVector vec_seq;
        for(int i =0; i<n;i++){
                vec_seq.push_back(i);
        }
        return vec_seq;
}


/// ====


// Defining the node class
class Node{

        public:
                // Storing parameters
                int index;
                Rcpp::NumericVector obs_train; // Training obs
                Rcpp::NumericVector obs_test; // Test obs
                int left; // Left node
                int right; // Right " "
                int depth;

                int var; // Variable which will be splitted
                double var_split; // Getting the variable that will be split

                double mu; // Getting mu parameter

                // Defining methods
                Node(int index_, Rcpp::NumericVector obs_train_,
                     Rcpp::NumericVector obs_test_,
                     int left_,int right_,
                     int depth_, int var_, double var_split_, double mu_){
                        index = index_;
                        obs_train = obs_train_;
                        obs_test = obs_test_;
                        left = left_;
                        right = right_;
                        depth = depth_;
                        var = var_;
                        var_split = var_split_;
                        mu = mu_;
                };
                void DisplayNode();
                bool isTerminal(){ return((left == -1) & (right == -1));};
                // double loglikelihood(arma::vec& residuals_values, double tau, double tau_mu);
                void update_mu(arma::vec& residuals_values, double tau, double tau_mu);
};



class Tree{

public:
        // Defining the main elements of the tree structure
        vector<Node> list_node;
        double t_log_likelihood;

        // Getting the vector of nodes
        Tree(int n_obs_train, int n_obs_test){
                list_node.push_back(Node(0,
                                    seq_along_cpp(n_obs_train),
                                    seq_along_cpp(n_obs_test),
                                    -1, // Left node
                                    -1, // Right node
                                    0, // Depth
                                    -1, // Var
                                    -1.1, // Var split
                                    0));

                // Tree loglikelihood
                t_log_likelihood = 0.0;
        }

        // Defining methods

        // Display methds
        void DisplayNodes();
        vector<Node> getTerminals();
        vector<Node> getNonTerminals();
        int n_terminal();
        int n_internal();
        int n_nog();

        // Main methods :
        // Getting the tree loglikelihood (BASED ON BARTMACHINES (BLEICH,2017))
        void new_tree_loglike(arma::vec& res_val,double tau, double tau_mu,Tree& current_tree,double& verb, int& id_node);
        // Update tree all the mu from all the trees
        void update_mu_tree(arma::vec& res_val, double tau, double tau_mu);
        void update_mu_tree_linero(arma::vec& res_val, double tau, double tau_mu,int& n_leaves, double& sq_mu_norm);
        // Growing a tree
        void grow(arma::vec res_values,
                  const arma::mat& x_train,const arma::mat& x_test,int node_min_size,const Rcpp::NumericMatrix& xcut,
                  double&tau, double& tau_mu, double& alpha, double& beta);
        // Creating the verb to prune a tree
        void prune(int& id_node);
        // Change a tree
        void change(const arma::mat& x_train,const arma::mat& x_test,int node_min_size,const arma::mat& xcut,int& id_t,int& id_node);
        // Function to calculate the tree prior loglikelihood
        double prior_loglilke(double alpha, double beta);
        // Updating the tree predictions
        void getPrediction(arma::vec &train_pred_vec,arma::vec &test_pred_vec);

};

// Functions to visualize the tree and nodes
void Node::DisplayNode(){

        std::cout << "Node Number: " << index << endl;
        std::cout << "Decision Rule -> Var:  " << var << " & Rule: " << var_split << endl;
        std::cout << "Left <-  " << left << " & Right -> " << right << endl;

        if(true){
                std::cout << "Observations train: " ;
                for(int i = 0; i<obs_train.size(); i++){
                        std::cout << obs_train[i] << " ";
                }
                std::cout << endl;
        }

        if(true){
                std::cout << "Observations test: " ;
                for(int i = 0; i<obs_test.size(); i++){
                        std::cout << obs_test[i] << " ";
                }
                std::cout << endl;
        }
                std::cout << " Tree split: " << var_split << endl;

}

// For trees
void Tree::DisplayNodes(){
        for(int i = 0; i<list_node.size(); i++){
                list_node[i].DisplayNode();
        }
        std::cout << "# ====== #" << endl;
}

// Getting terminal nodes
vector<Node> Tree::getTerminals(){

        // Defining terminal nodes
        vector<Node> terminalNodes;

        for(int i = 0; i<list_node.size(); i++){
                if(list_node[i].isTerminal()==1 ){ // Check this again, might remove the condition of being greater than 5
                        terminalNodes.push_back(list_node[i]); // Adding the terminals to the list
                }
        }
        return terminalNodes;
}


// Getting terminal nodes
vector<Node> Tree::getNonTerminals(){

        // Defining terminal nodes
        vector<Node> NonTerminalNodes;

        for(int i = 0; i<list_node.size(); i++){
                if(list_node[i].isTerminal()==0){
                        NonTerminalNodes.push_back(list_node[i]); // Adding the terminals to the list
                }
        }
        return NonTerminalNodes;
}


// Getting the number of n_terminals
int Tree::n_terminal(){

        // Defining the sum value
        int terminal_sum = 0;
        for(int i = 0; i<list_node.size(); i++){
                if(list_node[i].isTerminal()==1){
                        terminal_sum++;
                }
        }

        return terminal_sum;
}

// Getting the number of non-terminals
int Tree::n_internal(){

        // Defining the sum value
        int internal_sum = 0;
        for(int i = 0; i<list_node.size(); i++){
                if(list_node[i].isTerminal()==0){
                        internal_sum++;
                }
        }

        return internal_sum;
}

// Get the number of NOG (branches parents of terminal nodes)
int Tree::n_nog(){
        // Selecting possible parents of terminal nodes to prune
        vector<Node> nog_list;
        int nog_counter = 0;
        // Getting the parent of terminal node only
        for(int i = 0; i<list_node.size();i++){
                if(list_node[i].isTerminal()==0){
                        if(list_node[i+1].isTerminal()== 1 && list_node[i+2].isTerminal()==1){
                                nog_counter++;
                        }
                }
        }
        return nog_counter;
}


// Exporting the classes
RCPP_EXPOSED_CLASS(Node);
RCPP_EXPOSED_CLASS(Tree);
