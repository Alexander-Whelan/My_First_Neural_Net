
%% Neural Networks: Training of a Multilayer Perceptron
%ID:33589362
%
%Incremental backpropagation algorithm, no biases

clear all, close all, clc
%% Parameter Settings
learnrate=0.1
epochs=100
weights=[0.1;-0.3;-0.1;0.5;-0.4;0.1;0.2;-0.4;-0.2];

inputs=[1 0;1 1];
out_des=[0;1];

[n_examples n_inputs]=size(inputs);
n_nodes=3;
n_outputs=size(out_des, 2);

%% Weight Formatting
%all this code is just so that I can convert the example weights given as a
%column vector into a weight matrix where the index of the weight
%represents which input and node it is attached to. Any non-existent
%connection is assigned a zero.
weights_i2h=zeros(n_nodes, n_inputs);
weights_h2o=zeros(n_outputs, n_nodes);
weights_i2o=zeros(n_outputs, n_inputs);

weights_i2h(1,1)=weights(1);
weights_i2h(2,1)=weights(3);
weights_i2h(2,2)=weights(4);
weights_i2h(3,2)=weights(6);

weights_h2o(1,1)=weights(9);
weights_h2o(1,2)=weights(8);
weights_h2o(1,3)=weights(7);

weights_i2o(1,1)=weights(2);
weights_i2o(1,2)=weights(5);

for t=1:epochs
 for e=1:n_examples
%% Forward Pass

%input-to-hidden loop with sigmoid activation functions
for j=1:n_nodes
     for i=1:n_inputs
        hidden(j, i)=(weights_i2h(j, i))*(inputs(e, i));
     end
     hidden_out(j)=1/(1+exp(-(sum(hidden(j, :)))));
end

%hidden-to-out and direct input-to-out loops
for j=1:n_outputs
     for i=1:n_nodes
        out(j, i)=(weights_h2o(j, i))*(hidden_out(1, i));
     end
     for h=1:n_inputs
         directs(j, h)=(weights_i2o(j, h))*(inputs(e, h));
     end
     output(j)=sum([out(j, :), directs(j, :)]);
end
        
%% Backward Pass

%benefits at output and hidden nodes
error=out_des(e, :) - (output(1, :)); 
beta_out=error; %summation not sigmoid function at output node so beta(out) is equivalent to error
for i=1:n_nodes
    beta_hidden(i)=hidden_out(i) * (1 - hidden_out(i)) * (beta_out * weights_h2o(i));
end

%weight updates

%input-to-hidden loop
for j=1:n_inputs
    for i=1:n_nodes
        if weights_i2h(i, j) ~= 0
        weights_i2h(i, j)=weights_i2h(i, j) + (learnrate * beta_hidden(i) * inputs(e, j));
        else
        end
    end
end

%hidden-to-out loop
for i=1:n_nodes
     weights_h2o(i)=weights_h2o(i) + (learnrate * beta_out * hidden_out(1, i));
end

%input-to-out loop
for i=1:n_inputs
        weights_i2o(i)=weights_i2o(i) + (learnrate * beta_out * inputs(e, i));
end


errors(e) = error;
 end
pause(0.1)
TSS = (1/n_examples) * ((sum(errors))^2)
plotTSS(t)=TSS;
plot(plotTSS, 'LineWidth', 1, 'Color', [0.6 0.3 0.2], 'Marker', 'x')
xlabel('No. of Epochs')
ylabel('Mean Squared Error')
title('Neural Network Demonstration of Mean Squared Error Decrease')
hold on

end