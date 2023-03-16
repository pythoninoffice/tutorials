import pynecone as pc

def navbar(State):
    """The navbar."""
    return pc.box(
        pc.hstack(
            pc.link(
                pc.hstack(pc.image(src="favicon.ico"), pc.heading("Stock Chart")),
                href="/",
            ),
            pc.box(width='70vw'),
            pc.menu(
                pc.menu_button(
                    pc.cond(
                        State.logged_in,
                        pc.avatar(name=State.username, size="md"),
                        pc.box(),
                    )
                ),
                pc.menu_list(
                    pc.center(
                        pc.vstack(
                            pc.avatar(name=State.username, size="md"),
                            pc.text(State.username),
                        )
                    ),
                    pc.menu_divider(),
                    pc.link(pc.menu_item("Sign Out"),),
                ),
            ),
            pc.button(
                pc.icon(tag="moon"),
                on_click=pc.toggle_color_mode,
            ),
            justify="space-between",
            border_bottom="0.2em solid rgba(0,0,0,0.5)",
            padding_x="2em",
            padding_y="0.5em",
            #bg="rgba(255,255,255, 1)",
        ),
        position="fixed",
        width="100%",
        top="0px",
        z_index="500",
    )