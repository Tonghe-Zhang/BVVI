Run this command to check the shapes:

for h in range(H+1):
    print(sigma_hat[h].shape)

    torch.Size([4, 2])
    torch.Size([4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2])

for h in range(H+1):
    print(beta_hat[h].shape)

    torch.Size([4, 2])
    torch.Size([4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2, 4, 2])

for h in range(H):
    print(Q_function[h].shape)

    torch.Size([4, 2])
    torch.Size([4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2])

for h in range(H):
    print(value_function[h].shape)

    torch.Size([4])
    torch.Size([4, 2, 4])
    torch.Size([4, 2, 4, 2, 4])
    torch.Size([4, 2, 4, 2, 4, 2, 4])
    torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4])

test_policy():
    @ h=0, policy[0].shape=torch.Size([4, 2])
    @ h=1, policy[1].shape=torch.Size([4, 2, 4, 2])
    @ h=2, policy[2].shape=torch.Size([4, 2, 4, 2, 4, 2])
    @ h=3, policy[3].shape=torch.Size([4, 2, 4, 2, 4, 2, 4, 2])      
    @ h=4, policy[4].shape=torch.Size([4, 2, 4, 2, 4, 2, 4, 2, 4, 2])



