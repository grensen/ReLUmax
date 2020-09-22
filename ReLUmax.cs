using System;
using System.IO;

namespace ReLUmax
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("ReLUmax output demo with MNIST data");

            int input = 784; // MNIST image as input
            int output = 10; // 10 classes for each number
            int neuronLen = input + output;
            int weightLen = input * output;

            float learningRate = 0.005f;

            float[] neuron = new float[neuronLen], bias = new float[output];
            float[] weight = new float[weightLen], delta = new float[weightLen];

            // compare
            for (int main = 0, correct = 0; main < 2; main++, correct = 0)
            {      
                //--- testdata "t10k-images.idx3-ubyte" , "t10k-labels.idx1-ubyte" 
                FileStream MNISTlabels = new FileStream(@"C:\mnist\train-labels.idx1-ubyte", FileMode.Open);
                FileStream MNISTimages = new FileStream(@"C:\mnist\train-images.idx3-ubyte", FileMode.Open);
                MNISTimages.Seek(16, 0); MNISTlabels.Seek(8, 0);

                InitGlorot(123);

                for (int i = 0; i < weightLen; i++) delta[i] = 0;

                DateTime elapsed = DateTime.Now;
                if(main == 0)
                    Console.WriteLine("\n" + "Start traditional softmax approach\n");
                else
                    Console.WriteLine("\n" + "Start softmax modification with ReLU activation\n");

                //--- start training
                for (int x = 1; x < 60000 + 1; x++)
                {

                    //+----------- 1. Feed Input --------------------------------------------+      
                    for (int n = 0; n < input; ++n)
                        neuron[n] = MNISTimages.ReadByte() / 255.0f;
                    int target = MNISTlabels.ReadByte(), prediction = 0;

                    //+----------- 2. Feed Forward ------------------------------------------+     
                    float scale = 0, maxOut = float.MinValue, gradient = 0;
                    for (int k = 0; k < output; k++)
                    {
                        float net = 0; // bias[k];
                        for (int n = 0, w = k; n < input; n++, w += output)
                            net += neuron[n] * weight[w];
                    
                        if (net > maxOut) { maxOut = net; prediction = k; } // grab maxout here

                        if(main == 0)
                            scale += neuron[k + input] = (float)Math.Exp(net);  // more traditional approach
                        else
                            scale += neuron[k + input] = net > 0 ? (float)Math.Exp(net) : 0;  // net = net * net;                                                                                     //
                    }//--- k ends    

                    //+----------- 3. Backpropagation ---------------------------------------+    
                    for (int k = 0, j = neuronLen - 1; k != output; k++, j--) // output neuron loop
                        if ((gradient = output - k - 1 == target ? 1 - neuron[j] / scale : -neuron[j] / scale) != 0) // ugly?
                            for (int n = input - 1, wd = weightLen - k - 1; n > 0; wd -= output, n--)
                                delta[wd] += gradient * neuron[n]; // bias[k] += gradient * learningRate; // no need?

                    //+----------- 4. update Weights ----------------------------------------+         
                    if (target != prediction)
                        for (int m = 0; m < weightLen; m++)
                        {
                            weight[m] += learningRate * delta[m];
                            delta[m] *= 0.5f;
                        }

                    correct += target == prediction ? 1 : 0;

                    if (x % 10000 == 0)
                        Console.WriteLine("Iter = " + x + "   acc = " + (correct * 100.0f / x).ToString("F2") 
                            + "   batch = " + ((float)x /(x - correct)).ToString("F2"));
                } //--- runs end
                MNISTimages.Close(); MNISTlabels.Close();
                Console.WriteLine("\nCorrect = " + (correct) + "   incorrect = " + (60000 - correct));
                Console.WriteLine("Time = " + (((TimeSpan)(DateTime.Now - elapsed)).TotalMilliseconds / 1000.0).ToString("F2") + "s");
            } // main loop
            Console.WriteLine("\nEnd demo");
            Console.ReadLine();     

            void InitGlorot(int seed = 1)
            {
                Random rnd = new Random(seed);
                float sd = (float)Math.Sqrt(6.0f / (neuronLen));
                for (int m = 0; m < weightLen; m++) // weights
                    weight[m] = 2 * sd * (float)rnd.NextDouble() - sd;
            }
        } // main
    }
} // ns
