import simulation

if __name__ == '__main__':
    m2i_data_processed_by_mpc = simulation.run_mpc(r'20220921.pkl')
    print(m2i_data_processed_by_mpc['state/future/x'].shape)
    print(m2i_data_processed_by_mpc['state/future/velocity_x'].shape)