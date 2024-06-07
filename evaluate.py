def evaluate_model(model, test_images, timesteps):
    model.eval()
    with torch.no_grad():
        denoised_images = []
        for i in range(test_images.shape[0]):
            noisy_img = forward_diffusion(test_images[i:i+1], timesteps, alpha)[-1]
            denoised_img = backward_denoising(model, [noisy_img], timesteps)
            denoised_images.append(denoised_img)
        return np.array(denoised_images)

test_clean_images = load_nifti('test_clean_images.nii.gz')

denoised_images = evaluate_model(model, test_clean_images, timesteps)
print("Denoising completed.")

