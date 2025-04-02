    break
    
    train_data = create_dataloader(X_train, y_train, args)
    test_data = create_test_dataloader(X_test, y_test, args)
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    
    model = get_model(in_features=input_dim, out_features=output_dim, args=args, constraints_active=False)  
    if args.mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_model(model=model, train_data=train_data, test_data=test_data, optimizer=optimizer, input_dim=input_dim, output_dim=output_dim, stats=stats, X_test=X_test, y_test=y_test, args=args)
    
    model = get_model(in_features=input_dim, out_features=output_dim, args=args, constraints_active=True) 
    model.load_state_dict(torch.load('./models/'+args.model_id+'.pth') ['state_dict'])
    model.to(device)
    model.eval()
    create_report(model, X_test, y_test, stats, args)
    
if __name__ == "__main__":
    args = add_nn_arguments()
    main(args)