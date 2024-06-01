from gamma import gamma_index

def compute_gamma(reference, evaluation, dose_threshold=0.1, dose_diff=0.03, dist_mm=3.0):
    reference = reference.numpy()
    evaluation = evaluation.numpy()
    gamma_map = gamma_index(reference, evaluation, dose_threshold, dose_diff, dist_mm)
    return gamma_map


def evaluate_model(model, dataloader, device):
    model.eval()
    all_gamma_maps = []
    with torch.no_grad():
        for noisy_imgs, clean_imgs in dataloader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            outputs = model(noisy_imgs)

            for ref, eval in zip(clean_imgs, outputs):
                gamma_map = compute_gamma(ref.squeeze(), eval.squeeze())
                all_gamma_maps.append(gamma_map)

    return all_gamma_maps


# 加载测试数据
test_noisy_image_paths = ['path/to/test_noisy_image1.nii', 'path/to/test_noisy_image2.nii', ...]
test_clean_image_paths = ['path/to/test_clean_image1.nii', 'path/to/test_clean_image2.nii', ...]
test_dataset = CTNiftiDataset(test_noisy_image_paths, test_clean_image_paths, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 评估模型
gamma_maps = evaluate_model(model, test_dataloader, device)
mean_gamma = np.mean([np.mean(gamma_map) for gamma_map in gamma_maps])
print(f'Gamma 3%/3mm/10% mean value: {mean_gamma:.4f}')
