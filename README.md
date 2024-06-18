# Diffusion-study

### step()을 이용한 Diffusion 프로세스

- 핵심 코드 부분만 확인해보자.

```python
# 랜덤 정규분포 : N(0, 1)
x = torch.randn(4, 3, 256, 256).to(device)

for i, t in tqdm(enumerate(scheduler.timesteps)):
    # 이미지 전처리
    model_input = scheduler.scale_model_input(x, t)

    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]

    scheduler_output = scheduler.step(noise_pred, t, x)

    # Update x
    x = scheduler_output.prev_sample
```

- 여기서는 x의 값 범위를 (-1, 1)로 본다.
- 초기값으로 randn을 주더라도 크게 영향을 끼치지는 않는다.

```python
x = torch.randn(4, 3, 256, 256).to(device)
```

- scheduler.timesteps는 다음 값들을 가지고 있다.
- 최대 steps를 1000까지 나눌 수 있을 것이다.

```python
scheduler.timesteps
```

```
tensor([975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650,
        625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375, 350, 325, 300,
        275, 250, 225, 200, 175, 150, 125, 100,  75,  50,  25,   0])
```

- U-Net에 t라는 input이 추가되었는데 model_input의 noise 강도를 알려주어 학습 및 예측에 유리하다.
- 이제 output은 denoised된 이미지가 아닌 어떤 노이즈가 씌워졌는지에 대한 output이다.

```python
noise_pred = image_pipe.unet(model_input, t)["sample"]
```

- step과정에서 denoised된 pred_original_sample과 다음 스텝에 넣을 x인 prev_sample가 계산된다.
- 다음 스텝 x의 계산식은 이전과 다르지만 t가 0에 가까워질 수록 높은 denoise 비율을 적용한 다음 스텝 x가 계산 된다.

```python
scheduler_output = scheduler.step(noise_pred, t, x)
x = scheduler_output.prev_sample
```

![image](https://github.com/tetrapod0/Diffusion-study/assets/48349693/a4286675-e86f-496f-9c8e-8369cb93e157)
![image](https://github.com/tetrapod0/Diffusion-study/assets/48349693/067dbaab-9a35-4be1-adfb-8d500e6e4f6f)
![image](https://github.com/tetrapod0/Diffusion-study/assets/48349693/10be4d4b-aee5-4fd8-99db-f08ec620657c)
![image](https://github.com/tetrapod0/Diffusion-study/assets/48349693/515d607e-387d-44d9-8bd4-732cf2a77a1b)
![image](https://github.com/tetrapod0/Diffusion-study/assets/48349693/18ee614d-f4ae-4411-8d34-9ea15c20e2a1)
![image (1)](https://github.com/tetrapod0/Diffusion-study/assets/48349693/a455fd72-bea0-441b-9858-bda5793d1a41)













