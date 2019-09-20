function predY = tsvm_l(dataX,dataY,datapath,filename)
    
    example_file = strcat(datapath,strcat(filename,'.data'));
    model_file = strcat(datapath,strcat(filename,'.model'));
    output_file = strcat(datapath,strcat(filename,'.mat'));
    
    save(example_file,'dataX','dataY');
    svmlwrite(example_file,dataX,dataY);
    options=svmlopt();
    svm_learn(options,example_file,model_file);
    svm_classify(options,example_file,model_file,output_file);
    
    load(output_file,'-ascii');
    predY=pre;