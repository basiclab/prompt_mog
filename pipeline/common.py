def accumulative_concat(prompt: str) -> str:
    prompts = [prompt.strip() for prompt in prompt.split("<br>")]

    accum_list = [prompts[0]]
    cur_prompt = prompts[0]

    for prompt in prompts[1:]:
        accum_list.append(cur_prompt + " " + prompt)
        cur_prompt = accum_list[-1]

    return accum_list
