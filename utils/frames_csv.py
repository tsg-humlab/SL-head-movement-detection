def get_splits(df_frames):
    return sorted(set(df_frames[df_frames['split'].str.contains('fold')]['split']))


def get_n_splits(df_frames):
    return len(get_splits(df_frames))
