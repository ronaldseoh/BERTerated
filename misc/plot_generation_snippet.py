import matplotlib.pyplot as plt

plt.figure()
plt.xlabel("Rounds")
plt.ylabel("Loss")

plt.plot(tb_events_df['train_loss'], label='Training loss', c='orangered')

plt.plot(tb_events_df['validation_metric'], label='Validation loss', c='steelblue')

for i in range(50):
    client_losses = training_loop_client_loss.iloc[i]
    client_losses_nonzero = client_losses[client_losses > 0]
    plt.scatter([i+1] * len(client_losses_nonzero), client_losses_nonzero, c='silver')


plt.legend()

plt.savefig('Figures/plot.png')
plt.close()
